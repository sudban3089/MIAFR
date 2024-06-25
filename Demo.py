# Link: https://github.com/huggingface/diffusers/issues/3582#issuecomment-1568274591
#!/usr/bin/env python3
from diffusers import ControlNetModel, DDIMScheduler, DiffusionPipeline, StableDiffusionControlNetInpaintPipeline
import torch
import cv2
import numpy as np
import glob
import os
import sys
import pathlib
from pathlib import Path
from PIL import Image
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation


device = "cuda"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
     image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
     with torch.no_grad(), torch.autocast("cuda"):
         depth_map = depth_estimator(image).predicted_depth

     depth_map = torch.nn.functional.interpolate(
         depth_map.unsqueeze(1),
         size=(1024, 1024),
         mode="bicubic",
         align_corners=False,
     )
     depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
     depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
     depth_map = (depth_map - depth_min) / (depth_max - depth_min)
     image = torch.cat([depth_map] * 3, dim=1)
     image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
     image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
     return image




pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cpu").manual_seed(0)


# example prompts
#prompts = ["photo of a person with brown hair", "photo of a person with red lip", "photo of a person with blue eyes", "photo of a person wearing necktie","photo of a person with orange lips"]

img = load_image("./examples/1.jpg").resize((1024, 1024)) 
mask = load_image("./examples/1_lowerlipmask.png").resize((1024, 1024)) ## use the mask pertaining to the attribute to edit, use hair mask for changing hair color and it should correspond to the same subject for inpainting
depth_image = get_depth_map(img)

generator = torch.Generator(device="cpu").manual_seed(0) #torch.manual_seed(0) #

# generate image
image = pipe(
    "photo of a person with red lip",
    num_inference_steps=20,
    generator=generator,
    eta=1.0,
    image=img,
    mask_image=mask,
    control_image=depth_image,
).images[0]

save_path = './DemoOutputs'
os.makedirs(save_path,exist_ok=True)
image.save(f"{save_path}/Demo_1_redlowerlip.png")

