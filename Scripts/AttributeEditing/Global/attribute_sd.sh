#!/bin/bash
cd /path/to/yourscripts
source /ext3/env.sh
token=${1}
subject=${2}
data_path=${3}
dataset_name=${4}
config=config.yaml
reg=/path/to/CelebA_regularization_attribute
data=$data_path/$subject
name=/path/to/experiments_CelebAMaskHQ_attribute_reg/$dataset_name/$token/$subject
rm -rf $name
mkdir -p $name
chmod 777 -R $name
export token=$token
mkdir -p /results/CelebAMaskHQ_attribute_reg/$dataset_name/$token
chmod 777 -R /results/CelebAMaskHQ_attribute_reg/$dataset_name/$token
python main.py --base $config   -t --actual_resume v1-5-pruned.ckpt               -n $name              --gpus 0,                 --data_root $data                 --reg_data_root $reg                --class_word person                 --no-test 
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} young person" 
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} old person"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} male person"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt "photo of a ${token} female person"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} bald person"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with blackhair"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with blondhair"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with brownhair"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with bangs"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with nobeard"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with mustache"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with bushyeyebrows"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with slightlyopenmouth"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with doublechin"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with bignose"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with biglips"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with anger expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with happy expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with neutral expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with sad expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with surprise expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with fear expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person with disgust expression"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person wearing necktie"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person wearing hat"
python stable_txt2img.py --ddim_eta 0.0  --n_samples 4 --n_iter 1 --config $config --scale 10.0  --ddim_steps 100 --ckpt $name/checkpoints/last.ckpt  --outdir $name/images/inference --prompt  "photo of a ${token} person wearing eyeglasses"


rm -r $name/checkpoints/
rm -r $name/configs/
rm -r $name/testtube/
rm -r $name/images/train
rm -r $name/images/inference/samples
rm -r $name/images/inference/*.jpg
