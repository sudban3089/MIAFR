#!/bin/bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/scratch/sb9084/Dreambooth-Stable-Diffusion/IJCB/data/CelebA_TBIOM_rebuttal/20/TIDataset" #"./examples/20/SubID_0021_NumImages_20"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<ccorn-person>" \
  --initializer_token="person" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-03 \
  --scale_lr \
  --lr_scheduler="linear" \
  --lr_warmup_steps=500 \
  --num_vectors 5 \
  --output_dir="./textualinvembs/textual_inversion_ccorn_softmargin_numvect5linear"


