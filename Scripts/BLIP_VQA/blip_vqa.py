from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import glob
import os
from pathlib import Path

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


questions = ['Does the person have Arched Eyebrows?', 'Is the person Attractive?', 'Are there Bags Under Eyes?', ' Is the person Bald?','Are there hair Bangs?', 'Does the person have Big Lips?', 'Does the person have Big Nose?' , 'What is the hair color?', 'Are there Bushy Eyebrows?', 'Is the person Chubby?' ,  'Does the person have Double Chin?', 'Is the person wearing glasses?', 'Is there a Goatee?', 'Is the person wearing Heavy_Makeup?', 'Dioes the person have High Cheekbones?', 'What is the gender?', 'What is the age?', 'What is the ethnicity?', 'Is the Mouth Slightly Open?', 'Does the person have Mustache?', 'Does the person have Beard?', 'Are the eyes narrow?', 'Is the face shape Oval?', 'What is the emotional expression of the person in the photo?', 'What Accessories are present in the photo?']
destDir = "./BLIPcaptions_CelebAHQ" ## create your own directory for saving results
os.makedirs(f"{destDir}", exist_ok=True)

images = glob.glob("/path/to/CelebA-HQ-img/*.jpg")
for img in images:
    fname = Path(img).stem
    with open(img, 'rb') as file:
        image = Image.open(file)
        with open(os.path.join(destDir,f"{fname}.txt"), "w") as txtfile:
            for question in questions:
                prompt = f"Question: {question} Answer:" 
                inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                txtfile.write(generated_text+ "\n")
