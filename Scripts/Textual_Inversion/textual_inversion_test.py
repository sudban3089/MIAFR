from diffusers import StableDiffusionPipeline
import torch
import os

root_dir = "/scratch/sb9084/Dreambooth-Stable-Diffusion/IJCB/InstantID-main/outputs"
image_path = "specialtoken-person_textinv_numvect5"
file_path = os.path.join(root_dir,image_path)
os.makedirs(file_path, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion("path/to/trained/textualinversion/object or style concept")
image = pipeline("a high-resolution, detailed photo of <specialtoken-person> with brown hair", num_inference_steps=50, num_images_per_prompt=2).images
for ind, img  in enumerate(image):
    img.save(f"{file_path}/textinv_brownhairperson_{ind}.png")

# num_samples=1

# prompt3 = "a <subid0001-person> with blond hair" #@param {type:"string"}
# images3 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt5 = "a <subid0001-person> with necktie" #@param {type:"string"}
# images5 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]


# prompt7 = "a <subid0001-person> with bushy eyebrows" #@param {type:"string"}
# images7 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt8 = "a <subid0001-person> with sad expression" #@param {type:"string"}
# images8 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]


# images12.save(f"{file_path}/textinv_person_sd2_surprise.png")
    
