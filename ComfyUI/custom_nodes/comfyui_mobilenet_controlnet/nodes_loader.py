import json
import os

import torch
import folder_paths

from .mobilenet_control import MobileNetFluxControl


DTYPE_OPTIONS = ["default", "fp32", "bf16", "fp16"]

_SAFETENSORS_SUFFIX = ".safetensors"


def _is_mobilenet_safetensors(path):
    try:
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = f.read(min(header_len, 4_000_000))
        keys = json.loads(header).keys()
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    has_single = any(k.startswith("controlnet_single_blocks.") for k in keys)
    has_adapter = any(k.startswith("mobilenet_adapter.") for k in keys)
    return has_single and has_adapter


def get_mobilenet_controlnet_list():
    entries = []
    for base in folder_paths.get_folder_paths("controlnet"):
        if not os.path.isdir(base):
            continue
        for root, dirs, filenames in os.walk(base):
            if "config.json" in filenames:
                try:
                    with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
                        config = json.load(f)
                except (OSError, json.JSONDecodeError):
                    config = None
                if config is not None and config.get("_class_name") == "MobileNetFluxControlNetModel":
                    rel = os.path.relpath(root, base).replace(os.sep, "/")
                    entries.append(rel)
                    dirs[:] = []
                    continue
            for filename in filenames:
                if not filename.endswith(_SAFETENSORS_SUFFIX):
                    continue
                if not _is_mobilenet_safetensors(os.path.join(root, filename)):
                    continue
                rel_dir = os.path.relpath(root, base).replace(os.sep, "/")
                entries.append(filename if rel_dir == "." else f"{rel_dir}/{filename}")
    return sorted(set(entries))


def load_mobilenet_controlnet(path, dtype="default"):
    from .mobilenet_model import MobileNetFluxControlNetModel

    torch_dtype = None
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    if os.path.isfile(path) and path.endswith(_SAFETENSORS_SUFFIX):
        model = MobileNetFluxControlNetModel.from_single_file(path, torch_dtype=torch_dtype)
    else:
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = MobileNetFluxControlNetModel.from_pretrained(path, **kwargs)

    model.requires_grad_(False)
    model.eval()
    return MobileNetFluxControl(model)


class MobileNetFluxControlNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_mobilenet_controlnet_list()
        if not models:
            models = ["(no MobileNet Flux ControlNet folder found in models/controlnet/)"]
        return {
            "required": {
                "model_name": (models,),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders/controlnet"

    @staticmethod
    def _resolve(model_name):
        for base in folder_paths.get_folder_paths("controlnet"):
            candidate = os.path.join(base, *model_name.split("/"))
            if os.path.isdir(candidate) or (os.path.isfile(candidate) and candidate.endswith(_SAFETENSORS_SUFFIX)):
                return candidate
        return None

    def load_controlnet(self, model_name):
        path = self._resolve(model_name)
        if path is None:
            raise FileNotFoundError(f"MobileNet Flux ControlNet folder not found: {model_name}")
        return (load_mobilenet_controlnet(path),)


class MobileNetFluxControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "dtype": (DTYPE_OPTIONS, {"default": "default"}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet_advanced"
    CATEGORY = "loaders/controlnet"

    def load_controlnet_advanced(self, model_path, dtype="default"):
        model_path = os.path.expanduser(model_path.strip())
        if not (os.path.isdir(model_path) or (os.path.isfile(model_path) and model_path.endswith(_SAFETENSORS_SUFFIX))):
            raise FileNotFoundError(f"MobileNet Flux ControlNet file or folder not found at: {model_path}")
        return (load_mobilenet_controlnet(model_path, dtype),)


NODE_CLASS_MAPPINGS = {
    "MobileNetFluxControlNetLoader": MobileNetFluxControlNetLoader,
    "MobileNetFluxControlNetLoaderAdvanced": MobileNetFluxControlNetLoaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MobileNetFluxControlNetLoader": "Load MobileNet FLUX ControlNet",
    "MobileNetFluxControlNetLoaderAdvanced": "Load MobileNet FLUX ControlNet (Path)",
}
