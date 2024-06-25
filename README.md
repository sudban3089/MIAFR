# MIAFR
This is the official implementation of **Mitigating the Impact of Attribute Editing on Face Recognition** that has been accepted in International Joint Conference in Biometrics (IJCB 2024). Refer to [our paper](https://arxiv.org/html/2403.08092v1).

Run `Demo.py` for a quick demo to obtain the outputs below. Running the script may require sometime to load vae/diffusion-pytorch-model.safetensors and text_encoder/model.safetensors for the first time. You may enable the safety checker in the pipeline.

![alt text](GithubDemo.PNG)

## Overview
Through a large-scale study over diverse face images, we show that facial attribute editing using modern generative AI models can severely degrade automated face recognition systems. This degradation persists even with generative models that include additional identity-based loss function. To mitigate this issue, we propose two novel techniques for local and global attribute editing. We empirically ablate twenty-six facial semantic, demographic and expression-based attributes that have been edited using state-of-the-art generative models, and evaluate them using ArcFace and AdaFace matchers on CelebA, CelebAMaskHQ and LFW datasets. Finally, we use LLaVA, an emerging visual question-answering framework for attribute prediction to validate our editing techniques. Our methods outperform the current state-of-the-art at facial editing (BLIP, InstantID) while retaining identity by a significant extent.

![alt text](IJCB2024_Overview.PNG)

## Usage
- Create the `ldm` environment by following the steps outlined in [ID-Preserving-Facial-Aging](https://github.com/sudban3089/ID-Preserving-Facial-Aging) for global editing. We fine-tune a pre-trained stable diffusion model whose weights can be downloaded from [Hugging Face](https://huggingface.co/CompVis) model card. We use `v1-5-pruned.ckpt` for regularization-based fine-tuning for global editing.

## Fine-tuning
Fine-tuning is only required for DreamBooth-based global editing.

### Data preparation
We need a **Regularization Set** that comprises images depicting distinct individuals (disjoint from the training set) depicting varying attributes. We curated a set of 780 images that serves as image-caption pairs in this work. Download the Regularization Set used in our work from github or you can create your own regularization set but we cannot verify the performance with a custom regularization set. 

We need a **Training Set** that comprises images of a specific individual on whom the facial aging will be applied. The training set enables the diffusion model to learn the identity-specific charctristics during training that are then transferred at the time of generation of images with aging/de-aging. This repo currently supports single subject-specific training. You can create a custom batch script for training multiple subjects simultaneously, ensure that the rare token is linked to *each* subject uniquely, otherwise it may result in inconsistent outputs or identity lapse.   

We used the [contrastive loss](https://github.com/sudban3089/ID-Preserving-Facial-Aging) for fine-tuning.

### Facial aging/de-aging synthesis

After the completion of fine-tuning, you can generate photo on the trained individual by specifying the rare-token identifier (must match training), the `--class_word` and `age-label` denotes one of the six age groups indicated above.
```
python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --ckpt /path-to-saved-checkpoint-from-training
                                 --prompt "photo of a <rare-token> <class> as <age-label>"
                                 --outdir /directory to write output images
                                 --config /path to the configs file .yaml that was used during fine-tuning
```
### Evaluation

- We perform biometric matching (with ArcFace) using [deepface](https://github.com/serengil/deepface) library. We perform biometric matching using [AdaFace](https://github.com/mk-minchul/AdaFace)
- We use the official implementation of [BLIP Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion), [InstantID](https://github.com/InstantID/InstantID) and [MaskFaceGAN](https://github.com/MartinPernus/MaskFaceGAN) for baseline comparison.


## Citation
If you find this code useful or utilize it in your work, please cite:
```
@INPROCEEDINGS {MIAFR_IJCB2024,
author = {Sudipta Banerjee and Sai Pranaswi Mullangi and Shruti Wagle and Chinmay Hegde and Nasir Memon},
booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
title = {Mitigating the Impact of Attribute Editing on Face Recognition},
year = {2024},
}
```

## Acknowledgment and References
This repository is heavily dependent on code borrowed from different sources. 
 We use the official implementation of [ID-Preserving-Facial-Aging](https://github.com/sudban3089/ID-Preserving-Facial-Aging) for DB-prop. We use HuggingFace for implementing [textual inversion](https://huggingface.co/docs/diffusers/en/using-diffusers/textual_inversion_inference). We use the ControlNetV1.1 with Stable Diffusion V1.5 in [inpainting mode](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint). We use the ArcFace matcher from the [deepface library](https://github.com/serengil/deepface) and the [AdaFace matcher](https://github.com/mk-minchul/AdaFace) from its original implementation. We use the [BLIP Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion) from the LAVIS library. We use [LLaVA](https://huggingface.co/docs/transformers/main/en/model_doc/llava) and [BLIP-VQA](https://huggingface.co/Salesforce/blip-vqa-base) from the HuggingFace.


