#!/bin/sh
model_name=Crossformer
dataset_name=TBH
gpu_num=0
exo_input_len=52
num_meta=52
output_dim=512

for num_exo_vars in 2 3 5
do  
    for seed in 21 42 63 84 105
    do  
        python -u run.py \
        --seed $seed \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --gpu_num $gpu_num \
        --exo_input_len $exo_input_len \
        --num_meta $num_meta \
        --num_exo_vars $num_exo_vars \
        --output_dim $output_dim
    done
done