import torch

from .mobilenet_control import MobileNetFluxControl


class MobileNetFluxControlNetApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "model/conditioning/controlnet"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    base = control_net.copy() if isinstance(control_net, MobileNetFluxControl) else MobileNetFluxControl(control_net)
                    c_net = base.set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


NODE_CLASS_MAPPINGS = {
    "MobileNetFluxControlNetApply": MobileNetFluxControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MobileNetFluxControlNetApply": "Apply MobileNet FLUX ControlNet",
}
