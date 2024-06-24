from diffusers import StableDiffusionPipeline
import torch
import os

root_dir = "/scratch/sb9084/Dreambooth-Stable-Diffusion/IJCB/InstantID-main/outputs"
image_path = "bcorn-person_textinv_numvect5"
file_path = os.path.join(root_dir,image_path)
os.makedirs(file_path, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion("/scratch/sb9084/Dreambooth-Stable-Diffusion/IJCB/InstantID-main/textualinvembs/textual_inversion_bcorn_numvect5linear")
image = pipeline("a high-resolution, detailed photo of <bcorn-person> with long beard", num_inference_steps=50, num_images_per_prompt=2).images
for ind, img  in enumerate(image):
    img.save(f"{file_path}/textinv_beardperson_{ind}.png")

# num_samples=1

# prompt1 = "a <subid0001-person> " #@param {type:"string"}
# images1 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt2 = "a <subid0001-person> with blue hair" #@param {type:"string"}
# images2 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt3 = "a <subid0001-person> with blonde hair" #@param {type:"string"}
# images3 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt4 = "a <subid0001-person> as scared" #@param {type:"string"}
# images4 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt5 = "a <subid0001-person> with necktie" #@param {type:"string"}
# images5 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt6 = "a <subid0001-person> with long beard" #@param {type:"string"}
# images6 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt7 = "a <subid0001-person> with bushy eyebrows" #@param {type:"string"}
# images7 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt8 = "a <subid0001-person> as sad" #@param {type:"string"}
# images8 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt9 = "a <subid0001-person> as angry" #@param {type:"string"}
# images9 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt10 = "a <subid0001-person> as disgusted" #@param {type:"string"}
# images10 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt11 = "a <subid0001-person> as neutral" #@param {type:"string"}
# images11 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# prompt12 = "a <subid0001-person> as surprise" #@param {type:"string"}
# images12 = pipe(prompt1, num_images_per_prompt=num_samples, num_inference_steps=30, guidance_scale=7.5).images[0]

# root_dir = "/scratch/sb9084/Dreambooth-Stable-Diffusion/IJCB/InstantID-main/outputs"
# image_path = "SubID0001_textinv"
# file_path = os.path.join(root_dir,image_path)
# os.makedirs(file_path, exist_ok=True)

# images1.save(f"{file_path}/textinv_person_sd2.png")
# images2.save(f"{file_path}/textinv_person_sd2_bluehair.png")
# images3.save(f"{file_path}/textinv_person_sd2_blondehair.png")
# images4.save(f"{file_path}/textinv_person_sd2_scared.png")
# images5.save(f"{file_path}/textinv_person_sd2_necktie.png")
# images6.save(f"{file_path}/textinv_person_sd2_longbeard.png")
# images7.save(f"{file_path}/textinv_person_sd2_bushyeyebrows.png")
# images8.save(f"{file_path}/textinv_person_sd2_sad.png")
# images9.save(f"{file_path}/textinv_person_sd2_angry.png")
# images10.save(f"{file_path}/textinv_person_sd2_disgusted.png")
# images11.save(f"{file_path}/textinv_person_sd2_neutral.png")
# images12.save(f"{file_path}/textinv_person_sd2_surprise.png")
    
