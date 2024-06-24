
################# BLIP Diffusion Link: https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion ##############
### Blip Diffusion without ControlNet
# from diffusers.pipelines import BlipDiffusionPipeline
# from diffusers.utils import load_image
# import torch
# import os


# blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
#     "Salesforce/blipdiffusion", torch_dtype=torch.float16
# ).to("cuda")


# cond_subject = "person"
# tgt_subject = "person"
# text_prompt_input = "with black hair"

# cond_image = load_image(
#     "./examples/1.jpg"
# )
# guidance_scale = 7.5
# num_inference_steps = 25
# negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

# os.makedirs("./outputs/BLIPDiffusionoutputs/", exist_ok=True)
# output = blip_diffusion_pipe(
#     text_prompt_input,
#     cond_image,
#     cond_subject,
#     tgt_subject,
#     guidance_scale=guidance_scale,
#     num_inference_steps=num_inference_steps,
#     neg_prompt=negative_prompt,
#     height=512,
#     width=512,
# ).images
# output[0].save("./outputs/BLIPDiffusionoutputs/1_Blipblackhair.png")

### BLIP Diffusion with controlNet -- use this for better editability

from diffusers.pipelines import BlipDiffusionControlNetPipeline
from diffusers.utils import load_image
from controlnet_aux import CannyDetector
import torch
import glob
import os
import sys
import pathlib
from pathlib import Path
from PIL import Image

blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
    "Salesforce/blipdiffusion-controlnet", torch_dtype=torch.float16
).to("cuda")

style_subject = "person"
tgt_subject = "person"
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "blue eyes, over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
canny = CannyDetector()

startind = int(sys.argv[1])

prompts = ["with slightly open mouth", "with big lips", "with big nose", "with bushy eyebrows", "with double chin", "female", "with happy expression", "male", "wearing necktie", "with neutral expression", "with no beard", "old", "with sad expression", "with surprise expression", "young" ] #["wearing hat", "with bangs", "bald person", "with black hair", "with blonde hair", "with brown hair", "wearing necktie", "young person", "old person", "male person", "female person", "wearing eyeglasses", "with no beard", "with bushy eyebrows", "with mustache", "with double chin", "with big lips", "with big nose", "with slightly open mouth", "with anger expression", "with happy expression", "with neutral expression", "with sad expression", "with surprise expression", "with disgust expression", "with fear expression"]
# read image
image_path = 'path/to/CelebA_HQ_imgs'
orifiles = sorted(glob.glob(f'{image_path}/*.jpg')) 
print(len(orifiles))

refimgs_path = 'path/to/BLIPRefImgs' ## BLIPDiffusion requires a target/reference image for style transfer
reffiles = sorted(glob.glob(f'{refimgs_path}/*.jpg')) 
print(len(reffiles))

# save outputs
save_path = "/path/where/outputs/will/be/saved"
os.makedirs(save_path, exist_ok=True)

for ind, img in enumerate(orifiles[startind:], start=startind):
    print(ind)
    fname1= pathlib.PurePath(str(img))
    pathnm1 = fname1.stem
    cldm_cond_image = load_image(img).resize((512, 512))
    cldm_cond_image = canny(cldm_cond_image, 30, 70)
    for idx, prompt in enumerate(prompts):
        style_image = load_image(f'path/to/BLIPRefImgs/{idx}.jpg').resize((512, 512)) ## if you want to edit haorcolor of source image to black, plz provide a ref image whose hair is black. Refer to BLIPDiffusion code for details
        output = blip_diffusion_pipe(
            prompt,
            style_image,
            cldm_cond_image,
            style_subject,
            tgt_subject,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        ).images
        output[0].save(f"{save_path}/{pathnm1}_{prompt}.png")
