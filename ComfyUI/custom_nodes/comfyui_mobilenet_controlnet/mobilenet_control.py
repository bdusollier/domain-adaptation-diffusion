import torch

import comfy.controlnet
import comfy.model_management
import comfy.utils


class MobileNetFluxControl(comfy.controlnet.ControlBase):
    def __init__(self, control_model, device=None):
        super().__init__()
        self.control_model = control_model
        self.device = device if device is not None else comfy.model_management.get_torch_device()
        self.processed_input = None
        self.processed_shape = None

    @staticmethod
    def _token_grid(x_noisy):
        tokens_h = -(-x_noisy.shape[2] // 2)
        tokens_w = -(-x_noisy.shape[3] // 2)
        return tokens_h, tokens_w

    def _run_model(self, cond_hint, tokens_h, tokens_w):
        model = self.control_model
        model.to(device=self.device, dtype=cond_hint.dtype)
        try:
            out = model.compute_control(cond_hint, tokens_h, tokens_w)
        finally:
            model.cpu()
        return out

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = x_noisy.dtype
        tokens_h, tokens_w = self._token_grid(x_noisy)
        hint_batch = self.cond_hint_original.shape[0]
        shape_key = (hint_batch, tokens_h, tokens_w, dtype)

        if self.processed_input is None or self.processed_shape != shape_key:
            width = x_noisy.shape[3] * self.compression_ratio
            height = x_noisy.shape[2] * self.compression_ratio
            cond_hint = comfy.utils.common_upscale(self.cond_hint_original, width, height, self.upscale_algorithm, "center")
            if cond_hint.shape[1] > 3:
                cond_hint = cond_hint[:, :3]
            elif cond_hint.shape[1] == 1:
                cond_hint = cond_hint.repeat(1, 3, 1, 1)
            cond_hint = cond_hint.to(device=self.device, dtype=dtype)
            self.processed_input = self._run_model(cond_hint, tokens_h, tokens_w)
            del cond_hint
            self.processed_shape = shape_key

        target_batch = x_noisy.shape[0]
        control = {"input": [], "middle": [], "output": []}
        for key in ("input", "output"):
            for s in self.processed_input[key]:
                if s is None:
                    control[key].append(None)
                    continue
                broadcasted = comfy.controlnet.broadcast_image_to(s, target_batch, batched_number)
                control[key].append(broadcasted.clone() if broadcasted is s else broadcasted)
        return self.control_merge(control, control_prev, x_noisy.dtype)

    def copy(self):
        c = MobileNetFluxControl(self.control_model, device=self.device)
        self.copy_to(c)
        c.processed_input = None
        c.processed_shape = None
        return c

    def deepclone_multigpu(self, load_device, autoregister=False):
        import copy as _copy
        c = self.copy()
        c.control_model = _copy.deepcopy(c.control_model)
        c.device = load_device
        if autoregister:
            self.multigpu_clones[load_device] = c
        return c

    def cleanup(self):
        self.processed_input = None
        self.processed_shape = None
        super().cleanup()
