#!/bin/bash
echo "Process ID : $$"
echo "File name : $0"

#set -x

stage=$1
stop_stage=$2

project_path=/cpfs01/user/duyichao/workspace/LLM_RLAIF/MM-Reward
alignment_path=${project_path}/alignment
data_path=${alignment_path}/benchmark
image_path=${data_path}/images


model_list=(llava-v1.6-mistral-7b-hf idefics2-8b qwen llava-v1.6-mistral-7b-hf llava-v1.6-34b-hf)
axis_list=(missing attributes actions numbers position other)
data_source_list=(pap hpdv2 imagereward)

# for axis in "${axis_list[@]}"; do
#     for model in "${model_list[@]}"; do
#         python ${alignment_path}/get_rm_score_alignment.py \
#         --model ${model}  --config_path ${project_path}/config/config_alignment_eval.yaml \
#         --dataset  -l cache/ -s result/ -t 0.0
#     done
# done

CUDA_VISIBLE_DEVICES=0 \
python ${alignment_path}/get_rm_score_alignment.py \
    --model ${model_list[0]}  \
    --config_path ${project_path}/config/eval_alignment_config.yaml \
    --dataset ${data_path}/missing/pap_missing_test.json \
    --local_buffer ${image_path} \
    --save_dir ${alignment_path}/results/ \
    --threshold 0.0 --metric_scalar number