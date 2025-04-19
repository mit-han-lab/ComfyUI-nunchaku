from nunchaku.models.pulid.pipeline_flux import PuLIDPipeline
from nunchaku.models.pulid.pulid_forward import forward

from types import MethodType
from functools import partial
import torch
import numpy as np
from PIL import Image
class ApplyNunchakuPuLID:
    def __init__(self):
        self.pulid_device = "cuda"    
        self.weight_dtype = torch.bfloat16
        self.onnx_provider = "gpu"
        self.pretrained_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to encode"}),
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
                "ip_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "ip_weight"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Nunchaku"
    TITLE = "Apply Nunchaku PuLID"

    def apply(self, image, model, ip_weight):
        self.pulid_model = PuLIDPipeline(
            dit=model.model.diffusion_model.model,
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
        )
        self.pulid_model.load_pretrain(self.pretrained_model)
        image = image.squeeze().cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8) 
        id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(image)
        model.model.diffusion_model.model.forward = MethodType(partial(forward, id_embeddings=id_embeddings, id_weight=ip_weight), model.model.diffusion_model.model)
        return (model,)


