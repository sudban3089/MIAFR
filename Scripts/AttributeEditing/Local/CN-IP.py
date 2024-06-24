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


## args

startind = int(sys.argv[1]) # start index of the filename of image you want to edit
attribute = sys.argv[2] # name of attribute you would like to edit

device = "cuda"

#controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16) ## uncomment to use canny map as control input
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16) ## uncomment to use depth map as control input

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
#feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas") ## this is deprecated in some versions
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

# read image
image_path = '/path/to/CelebA_HQ_imgs'
orifiles = sorted(glob.glob(f'{image_path}/*.jpg')) 
print(len(orifiles))

# read mask
maskfiles =[]
mask_path = '/path/to/Celeba_HQ_masks'
for path in Path(f"{mask_path}").glob('*.png'):
    fname =path.stem
    #print(str(fname))
    if attribute in str(fname):
        maskfiles.append(str(path))
print(len(maskfiles))

# save outputs
save_path = "/scratch/sb9084/Dreambooth-Stable-Diffusion/IJCB/InstantID-main/IJCBOutputs"
os.makedirs(save_path, exist_ok=True)

## prompts
#prompts = ["photo of a person with black hair", "photo of a person with blonde hair", "photo of a person with brown hair", " ", "photo of a person with bangs", "photo of a bald person"]
#prompts = ["photo of a person wearing hat", "photo of a person not wearing hat", " "]
#prompts = ["photo of a person with bangs"]mustache
#prompts = ["photo of a young person", "photo of a old person", "photo of a male person", "photo of a female person", "photo of a person wearing eyeglasses", "photo of a person with no beard", "photo of a person with bushy eyebrows", "photo of a person with mustache", "photo of a person with double chin", "photo of a person with big lips", "photo of a person with big nose", "photo of a person with slightly open mouth", "photo of a person with anger expression", "photo of a person with happy expression", "photo of a person with neutral expression", "photo of a person with sad expression", "photo of a person with surprise expression", "photo of a person with disgust expression", "photo of a person with fear expression"]
#prompts = ["photo of a person wearing necktie"]
prompts = ["photo of a person with blue eyes"]

for ind, img in enumerate(orifiles[startind:], start=startind):
    print(ind)
    fname1= pathlib.PurePath(str(img))
    pathnm1 = fname1.stem
    faceimg = load_image(img).resize((1024,1024))
    depthimg = get_depth_map(faceimg)
    #print(pathnm1)
    for mask in maskfiles:
        fname2= pathlib.PurePath(str(mask))
        pathnm2 = fname2.stem
        split= pathnm2.split('_')
        if int(split[0]) == int(pathnm1):
            facemask = load_image(mask).resize((1024,1024))  
            for prompt in prompts:
            # generate image
                image = pipe(
                    prompt,
                    num_inference_steps=20,
                    generator=generator,
                    eta=1.0,
                    image=faceimg,
                    mask_image=facemask,
                    control_image=depthimg,
                ).images[0]
                image.save(f"{save_path}/{pathnm1}_{prompt}_botheyesmask.png")

