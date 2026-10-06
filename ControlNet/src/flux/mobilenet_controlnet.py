import os
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.modeling_utils import ModelMixin

from .mobilenet_adapter import MobileNetT2IAdapter


logger = logging.get_logger(__name__)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class LoRAZeroLinear(nn.Module):
    """
    Projection linéaire LoRA factorisée avec initialisation zéro.
    Permet une projection efficace O(B*N*r*(d_in + d_out)) vers chaque bloc FLUX.
    """
    def __init__(self, in_features=768, out_features=3072, r=32, alpha=32):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


@dataclass
class FluxControlNetOutput(BaseOutput):
    controlnet_block_samples: tuple[torch.Tensor]
    controlnet_single_block_samples: tuple[torch.Tensor]


class MobileNetFluxControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    Modèle ControlNet pour FLUX.1 fondé sur l'adaptateur multi-échelle MobileNetV3-Small
    et des projections LoRA factorisées vers les 19 blocs dual-stream et 38 blocs single-stream.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        flux_hidden_dim: int = 3072,
        guidance_embeds: bool = True,
        mobilenet_num_in_channels: int = 3,
        lora_rank: int = 32,
        lora_alpha: int = 32,
    ):
        super().__init__()

        self.mobilenet_adapter = MobileNetT2IAdapter(
            flux_hidden_dim=flux_hidden_dim,
            num_in_channels=mobilenet_num_in_channels,
        )

        self.controlnet_blocks = nn.ModuleList([
            LoRAZeroLinear(in_features=768, out_features=flux_hidden_dim, r=lora_rank, alpha=lora_alpha)
            for _ in range(num_layers)
        ])

        self.controlnet_single_blocks = nn.ModuleList([
            LoRAZeroLinear(in_features=768, out_features=flux_hidden_dim, r=lora_rank, alpha=lora_alpha)
            for _ in range(num_single_layers)
        ])

        self.gradient_checkpointing = False

    @property
    def input_hint_block(self):
        return self.mobilenet_adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> FluxControlNetOutput | tuple:
        if controlnet_cond.ndim == 4:
            latent_h = controlnet_cond.shape[-2] // 16
            latent_w = controlnet_cond.shape[-1] // 16
        else:
            latent_h = int(math.sqrt(hidden_states.shape[1]))
            latent_w = hidden_states.shape[1] // latent_h

        adapter_features = self.mobilenet_adapter(controlnet_cond, latent_h, latent_w)

        controlnet_block_samples = tuple(block(adapter_features) for block in self.controlnet_blocks)
        controlnet_single_block_samples = tuple(
            block(adapter_features) for block in self.controlnet_single_blocks
        )

        if conditioning_scale != 1.0:
            controlnet_block_samples = tuple(sample * conditioning_scale for sample in controlnet_block_samples)
            controlnet_single_block_samples = tuple(
                sample * conditioning_scale for sample in controlnet_single_block_samples
            )

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)

        adapter_state_dict = self.mobilenet_adapter.state_dict()
        torch.save(adapter_state_dict, f"{save_directory}/mobilenet_adapter.bin")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        controlnet = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        adapter_path = f"{pretrained_model_name_or_path}/mobilenet_adapter.bin"
        if os.path.isfile(adapter_path):
            try:
                adapter_state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
            except TypeError:
                adapter_state_dict = torch.load(adapter_path, map_location="cpu")
            controlnet.mobilenet_adapter.load_state_dict(adapter_state_dict, strict=False)

        return controlnet
