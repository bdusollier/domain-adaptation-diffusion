import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

logger = logging.getLogger(__name__)


class LoRAZeroLinear(nn.Module):
    def __init__(self, in_features: int = 768, out_features: int = 3072, r: int = 32, alpha: int | None = None):
        super().__init__()
        self.r = r
        if alpha is None:
            alpha = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class MobileNetT2IAdapter(nn.Module):
    def __init__(self, flux_hidden_dim: int = 3072, num_in_channels: int = 3, adapter_dim: int = 768):
        super().__init__()
        self.flux_hidden_dim = flux_hidden_dim
        self.adapter_dim = adapter_dim

        base_mobilenet = mobilenet_v3_small(weights=None)

        if num_in_channels != 3:
            old_conv = base_mobilenet.features[0][0]
            base_mobilenet.features[0][0] = nn.Conv2d(
                num_in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.features = base_mobilenet.features
        self.conv_in = nn.Conv2d(576, 512, kernel_size=1)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, adapter_dim, kernel_size=1)

    def forward(self, condition_image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        feat = self.features(condition_image)
        feat = self.conv_in(feat)
        feat = self.act(feat)
        feat = self.conv_out(feat)

        feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)

        B, C, H, W = feat.shape
        adapter_tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return adapter_tokens


class MultiScaleMobileNetT2IAdapter(nn.Module):
    def __init__(self, flux_hidden_dim: int = 3072, num_in_channels: int = 3, adapter_dim: int = 768):
        super().__init__()
        self.flux_hidden_dim = flux_hidden_dim
        self.adapter_dim = adapter_dim

        base_mobilenet = mobilenet_v3_small(weights=None)

        if num_in_channels != 3:
            old_conv = base_mobilenet.features[0][0]
            base_mobilenet.features[0][0] = nn.Conv2d(
                num_in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.stage1 = base_mobilenet.features[:4]
        self.stage2 = base_mobilenet.features[4:9]
        self.stage3 = base_mobilenet.features[9:12]
        self.stage4 = base_mobilenet.features[12:]

        self.lat1 = nn.Conv2d(24, 128, kernel_size=1)
        self.lat2 = nn.Conv2d(48, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(96, 256, kernel_size=1)
        self.lat4 = nn.Conv2d(576, 256, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 128 + 256 + 256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, adapter_dim, kernel_size=1),
        )

    def forward(self, condition_image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        x1 = self.stage1(condition_image)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        p1 = F.interpolate(self.lat1(x1), size=(target_h, target_w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.lat2(x2), size=(target_h, target_w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.lat3(x3), size=(target_h, target_w), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.lat4(x4), size=(target_h, target_w), mode="bilinear", align_corners=False)

        feat = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))

        B, C, H, W = feat.shape
        adapter_tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return adapter_tokens


class MobileNetFluxControlNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        flux_hidden_dim: int = 3072,
        guidance_embeds: bool = True,
        mobilenet_num_in_channels: int = 3,
        adapter_type: str = "single_scale",
        adapter_dim: int = 768,
        lora_rank: int = 32,
    ):
        super().__init__()

        if adapter_type == "multi_scale":
            self.mobilenet_adapter = MultiScaleMobileNetT2IAdapter(
                flux_hidden_dim=flux_hidden_dim,
                num_in_channels=mobilenet_num_in_channels,
                adapter_dim=adapter_dim,
            )
        else:
            self.mobilenet_adapter = MobileNetT2IAdapter(
                flux_hidden_dim=flux_hidden_dim,
                num_in_channels=mobilenet_num_in_channels,
                adapter_dim=adapter_dim,
            )

        self.controlnet_blocks = nn.ModuleList([
            LoRAZeroLinear(in_features=adapter_dim, out_features=flux_hidden_dim, r=lora_rank)
            for _ in range(num_layers)
        ])

        self.controlnet_single_blocks = nn.ModuleList([
            LoRAZeroLinear(in_features=adapter_dim, out_features=flux_hidden_dim, r=lora_rank)
            for _ in range(num_single_layers)
        ])

    def compute_control(
        self,
        controlnet_cond: torch.Tensor,
        latent_h: int,
        latent_w: int,
        conditioning_scale: float = 1.0,
    ) -> dict:
        adapter_features = self.mobilenet_adapter(controlnet_cond, latent_h, latent_w)

        controlnet_block_samples = [block(adapter_features) for block in self.controlnet_blocks]
        controlnet_single_block_samples = [
            block(adapter_features) for block in self.controlnet_single_blocks
        ]

        if conditioning_scale != 1.0:
            controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
            controlnet_single_block_samples = [
                sample * conditioning_scale for sample in controlnet_single_block_samples
            ]

        return {
            "input": controlnet_block_samples,
            "middle": [],
            "output": controlnet_single_block_samples,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        latent_size: tuple[int, int] | None = None,
        **kwargs,
    ) -> tuple:
        if latent_size is not None:
            latent_h, latent_w = latent_size
        else:
            latent_h = int(math.sqrt(hidden_states.shape[1]))
            latent_w = hidden_states.shape[1] // latent_h

        out = self.compute_control(controlnet_cond, latent_h, latent_w, conditioning_scale)
        return (tuple(out["input"]), tuple(out["output"]))

    @classmethod
    def _build_from_state_dict(cls, state_dict: dict, torch_dtype: torch.dtype | None = None):
        num_layers = len({k.split(".")[1] for k in state_dict if k.startswith("controlnet_blocks.")})
        num_single_layers = len({k.split(".")[1] for k in state_dict if k.startswith("controlnet_single_blocks.")})
        flux_hidden_dim = state_dict["controlnet_blocks.0.lora_B"].shape[0]

        is_multi_scale = any(k.startswith("mobilenet_adapter.stage1.") for k in state_dict)
        if is_multi_scale:
            adapter_type = "multi_scale"
            mobilenet_num_in_channels = state_dict["mobilenet_adapter.stage1.0.0.weight"].shape[1]
            adapter_dim = state_dict["mobilenet_adapter.fusion.2.weight"].shape[0]
        else:
            adapter_type = "single_scale"
            mobilenet_num_in_channels = state_dict["mobilenet_adapter.features.0.0.weight"].shape[1]
            adapter_dim = state_dict["mobilenet_adapter.conv_out.weight"].shape[0]

        if "controlnet_blocks.0.lora_A" in state_dict:
            r = state_dict["controlnet_blocks.0.lora_A"].shape[0]
        elif "controlnet_single_blocks.0.lora_A" in state_dict:
            r = state_dict["controlnet_single_blocks.0.lora_A"].shape[0]
        else:
            r = 32

        controlnet = cls(
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            flux_hidden_dim=flux_hidden_dim,
            mobilenet_num_in_channels=mobilenet_num_in_channels,
            adapter_type=adapter_type,
            adapter_dim=adapter_dim,
            lora_rank=r,
        )
        controlnet.load_state_dict(state_dict, strict=True)

        if torch_dtype is not None:
            controlnet = controlnet.to(torch_dtype)
        return controlnet

    @classmethod
    def from_single_file(cls, path: str, torch_dtype: torch.dtype | None = None):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(path)
        return cls._build_from_state_dict(state_dict, torch_dtype=torch_dtype)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, torch_dtype: torch.dtype | None = None, **kwargs):
        import safetensors.torch
        weights_path = None
        for cand in [
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
        ]:
            p = os.path.join(pretrained_model_name_or_path, cand)
            if os.path.isfile(p):
                weights_path = p
                break

        if weights_path is not None:
            state_dict = safetensors.torch.load_file(weights_path)
        else:
            bin_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.bin")
            if os.path.isfile(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(f"No weights file found in {pretrained_model_name_or_path}")

        adapter_path = os.path.join(pretrained_model_name_or_path, "mobilenet_adapter.bin")
        if os.path.isfile(adapter_path):
            adapter_sd = torch.load(adapter_path, map_location="cpu", weights_only=True)
            for k, v in adapter_sd.items():
                full_k = f"mobilenet_adapter.{k}" if not k.startswith("mobilenet_adapter.") else k
                state_dict[full_k] = v

        return cls._build_from_state_dict(state_dict, torch_dtype=torch_dtype)
