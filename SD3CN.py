import torch
import os
import folder_paths
from diffusers import StableDiffusion3Pipeline
from diffusers.models.controlnet_sd3 import ControlNetSD3Model
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

from .pipeline_stable_diffusion_3_controlnet import StableDiffusion3CommonPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class SD3MCN_BaseModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("STRING", {"default": "v2ray/stable-diffusion-3-medium-diffusers"}),
                "cn_model": (["InstantX/SD3-Controlnet-Canny", "InstantX/SD3-Controlnet-Pose", "InstantX/SD3-Controlnet-Tile", "InstantX/SD3-Controlnet-Inpainting"],),
            }
        }

    RETURN_TYPES = ("SD3MMODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "🖼️SD3MCN"
  
    def load_model(self, base_model, cn_model):

        cn_model = [cn_model]
        
        pipe = StableDiffusion3CommonPipeline.from_pretrained(
            base_model,
            controlnet_list=cn_model,
        ).to(device, dtype=torch.float16)
        return [pipe]


class SD3MCN_Generation_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("SD3MMODEL",),
                "image": ("IMAGE",), 
                "positive": ("STRING", {"default": "cat", "multiline": True}),
                "negative": ("STRING", {"default": "worst quality, low quality", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}), 
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 9, "min": 0, "max": 10}),
                "control_weight": ("FLOAT", {"default": 0.6, "min": 0, "max": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "🖼️SD3MCN"
                       
    def generate_image(self, pipe, image, positive, negative, steps, seed, width, height, guidance_scale, control_weight):

        generator = torch.Generator(device=device).manual_seed(seed)
      
        controlnet_conditioning = [
            dict(
                control_index=0,
                control_image=tensor2pil(image),
                control_weight=control_weight,
                control_pooled_projections='zeros'
            )
        ]

        controlnet_start_step = int(steps * 0.0)
        controlnet_end_step = int(steps * 1.0)
        
        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning=controlnet_conditioning,
            controlnet_start_step=controlnet_start_step,
            controlnet_end_step=controlnet_end_step,
            generator=generator,
            latents=None,
        )[0]

        output_t = pil2tensor(output)
        output_t = output_t.squeeze(0)
        print(output_t.shape)
        
        return (output_t,)


NODE_CLASS_MAPPINGS = {
    "SD3MCN_BaseModelLoader_Zho": SD3MCN_BaseModelLoader_Zho,
    "SD3MCN_Generation_Zho": SD3MCN_Generation_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3MCN_BaseModelLoader_Zho": "🖼️SD3MCN ModelLoader Zho",
    "SD3MCN_Generation_Zho": "🖼️SD3MCN Generation Zho"
}
