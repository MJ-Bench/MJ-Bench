#!/bin/bash

# perspective=$1 
# reward_model_name=$2 
perspective=artifacts 
reward_model_name=idefics2-8b
metric_type=number
metric_scale=10

project_path=/cpfs01/user/duyichao/workspace/LLM_RLAIF/MRM-Bench

pretrained_model=/cpfs01/user/duyichao/models/runwayml/stable-diffusion-v1-5
config_path=${project_path}/config/config_rm.yaml
prompt_file=${project_path}/finetune_datasets/${perspective}_finetune_prompt.txt
reward_model_prompt_file=${project_path}/eval/prompts_single_image/${perspective}_single_${metric_type}_scale${metric_scale}.txt
save_dir=${project_path}/result/finetune/${perspective}/${reward_model_name}

mkdir -p ${save_dir}/log

python3 ${project_path}/experimental/ddpo_finetune_with_vlm.py \
    --pretrained_model ${pretrained_model} \
    --reward_model_name ${reward_model_name} \
    --config_path ${config_path} \
    --prompt_file ${prompt_file} \
    --reward_model_prompt_file ${prompt_file} \
    --num_epochs 50 \
    --train_gradient_accumulation_steps 1 \
    --sample_num_steps 64 \
    --sample_batch_size 64 \
    --train_batch_size 64 \
    --mixed_precision fp16 \
    --sample_num_batches_per_epoch 64 \
    --per_prompt_stat_tracking True \
    --per_prompt_stat_tracking_buffer_size 4 \
    --tracker_project_name stable_diffusion_training \
    --use_lora True   \
    --save_dir ${save_dir} \
    --perspective ${perspective}  --metric_scale 10 --reward_model_device "cuda:1" \
    >${save_dir}/log/${perspective}_${reward_model_name}_ddpo_ft.log 2>&1

    # --log_with "wandb" \
