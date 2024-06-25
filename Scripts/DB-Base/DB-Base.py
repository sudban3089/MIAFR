#Import all the libraries 
import json
import os
import shutil
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display
import subprocess

#create instance_data_dir and dump concepts_list.json

concepts_list_celeba_sdv1 = [
    {
        "instance_prompt":      "photo of zwx person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    "/path/to/original_images",
        "class_data_dir":       "/path/to/regularization_set"
    },
]
for c in concepts_list_celeba_sdv1:
    os.makedirs(c["instance_data_dir"], exist_ok=True)
with open("concepts_list_celeba_sdv1.json", "w") as f:
    json.dump(concepts_list_celeba_sdv1, f, indent=4)

#Diffusion weights. Please, visit the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), read the license and tick the checkbox if you agree. You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work.
# https://huggingface.co/settings/tokens

# Define the directory path
huggingface_directory = os.path.expanduser("~/.huggingface")
# Create the directory if it doesn't exist
os.makedirs(huggingface_directory, exist_ok=True)
# Define the HUGGINGFACE_TOKEN
# Here give your own huggingface token
HUGGINGFACE_TOKEN = "{HUGGINGFACE_TOKEN}"
# Define the path for the token file
token_file_path = os.path.join(huggingface_directory, "token")
# Write the token to the file
with open(token_file_path, "w") as token_file:
    token_file.write(HUGGINGFACE_TOKEN)


#Model Name
MODEL_NAME = "runwayml/stable-diffusion-v1-5" #@param {type:"string"}

#Run the loop for 100 subjects

for i in range(100):
    source_directory = "/path/to/each/subfolder/in/instance_data_dir/for/each/iteration"
    destination_directory = "/path/to/folder/where/instance_data_dir/images/will/be/moved"
    # List all files in the source directory
    files_to_copy = os.listdir(source_directory)
    # copy each file from the source directory to the destination directory to ease programming 
    for file_name in files_to_copy:
        #print("fn",file_name)
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(destination_directory, file_name)
        # Check if the path is a file (not a subdirectory) before moving
        if os.path.isfile(source_path):
            shutil.copy(source_path, destination_path)
            print(f"Copied {file_name} to {destination_directory}")
        else:
            print(f"{file_name} is a subdirectory and won't be moved.")

    #this path is where weights of a particular subject will be stored 
    OUTPUT_DIR = "sd_1_5/sub_{}/zwx".format(i+1)
    OUTPUT_DIR = "/path/to/folder/where/OUTPUT_DIR/in/previous_step/is/located" + OUTPUT_DIR
    print(f"[*] Weights will be saved at {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Training
    shell_command = (
        "python3 /path/to/train_dreambooth.py/ "
        "--pretrained_model_name_or_path={} "
        "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse "
        "--output_dir={} "
        "--revision=fp16 "
        "--with_prior_preservation --prior_loss_weight=1.0 "
        "--seed=1337 "
        "--resolution=512 "
        "--train_batch_size=1 "
        "--train_text_encoder "
        "--mixed_precision=fp16 "
        "--use_8bit_adam "
        "--gradient_accumulation_steps=1 "
        "--learning_rate=1e-6 "
        "--lr_scheduler=constant "
        "--lr_warmup_steps=0 "
        "--num_class_images=600 "
        "--sample_batch_size=4 "
        "--max_train_steps=800 "
        "--save_interval=10000 "
        '--save_sample_prompt="photo of zwx person" '
        "--concepts_list=concepts_list_celeba_sdv1.json"
        ).format(MODEL_NAME, OUTPUT_DIR)
    #Execute the shell command
    subprocess.run(shell_command, shell=True)
    WEIGHTS_DIR = "" #@param {type:"string"}
    if WEIGHTS_DIR == "":
        #print("w",natsorted(glob(OUTPUT_DIR + os.sep + "*")))
        WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
        print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")
    
    #Inference
    model_path = WEIGHTS_DIR  # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = None
    
    #Can set random seed here for reproducibility.
    g_cuda = torch.Generator(device='cuda')
    seed = 52362 #@param {type:"number"}
    g_cuda.manual_seed(seed)

    # Run for generating images.
    prompt_dict = {
        "1":["photo of zwx person","no_attribute"],
        "2":["photo of zwx person smiling","smiling"],
        "3":["photo of zwx young person","young"],
        "4":["photo of zwx old person","old"],
        "5":["photo of zwx male person","male"],
        "6":["photo of zwx female person","female"],
        "7":["photo of zwx bald person","bald"],
        "8":["photo of zwx person with black hair", "black hair"],
        "9":["photo of zwx person with brown hair","brown hair"],
        "10":["photo of zwx person with bangs", "bangs"],
        "11":["photo of zwx person with blond hair","blond hair"],
        "12":["photo of zwx person wearing eyeglasses","eyeglasses"],
        "13":["photo of zwx person with no beard","no beard"],
        "14":["photo of zwx person with mustache","mustache"],
        "15":["photo of zwx person with bushy eyebrows","bushy eyebrows"],
        "16":["photo of zwx person with slightly open mouth","open mouth"],
        "17":["photo of zwx person wearing necktie","necktie"],
        "18":["photo of zwx person wearing hat","hat"],
        "19":["photo of zwx person with double chin","double chin"],
        "20":["photo of zwx person with big nose","big nose"],
        "21":["photo of zwx person with big lips","big lips"],
        "22":["photo of zwx angry person","angry"],
        "23":["photo of zwx person with neutral expression","neutral"],
        "24":["photo of zwx person with surprise expression","surprise"],
        "25":["photo of zwx person with sad expression","sad"],
        "26":["photo of zwx person with disgust expression","disgust"],
        "27":["photo of zwx person with fear expression","fear"]
    }
    for j in prompt_dict.values():
        prompt = j[0] #@param {type:"string"}
        negative_prompt = "" #@param {type:"string"}
        num_samples = 4 #@param {type:"number"}
        guidance_scale = 7.5 #@param {type:"number"}
        num_inference_steps = 24 #@param {type:"number"}
        height = 512 #@param {type:"number"}
        width = 512 #@param {type:"number"}

        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images
        target_folder =  '/path/to/destination/folder/where/images/will/be/generated'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        r = 1
        for img in images:
            #display(img)
            #colab_path is used to copy generated files to target_folder with a different name containing sub id and sample id
            colab_path = '/path/to/each/image/file/in/generated/images/for/each/iteration'
            img.save(colab_path)
            drive_path = os.path.join(target_folder,'Sub{0:0=4d}_Sample{1:0=4d}_{2}.png').format(i+1,r,j[1]) 
            shutil.copy(colab_path,drive_path)
            r=r+1
    #delete diffuser or files
    #Delete diffuser and old weights and only keep the ckpt to free up drive space.
    # Caution, Only execute if you are sure u want to delete the diffuser format weights and only use the ckpt.

    for f in glob(OUTPUT_DIR+os.sep+"*"):
        if f != WEIGHTS_DIR:
            shutil.rmtree(f)
            print("Deleted", f)
    for f in glob(WEIGHTS_DIR+"/*"):
        if not f.endswith(".json"):
            try:
                shutil.rmtree(f)
            except NotADirectoryError:
                continue
            print("Deleted", f)
    #delete output_dir/800 folder
    # for f in glob(OUTPUT_DIR+"/*"):
    #  try:
    #   shutil.rmtree(f)
    # except NotADirectoryError:
    #   continue
    # print("Deleted", f)
    #delete instance_dir images
    folder_path = "/path/same/as/destination_directory/above"
    # List all files in the folder
    files_to_delete = os.listdir(folder_path)
    # Delete each file in the folder
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        # Check if the path is a file (not a subdirectory) before deleting
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_name} from {folder_path}")
        else:
            print(f"{file_name} is a subdirectory and won't be deleted.")
