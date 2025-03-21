model_name=iTransformer
dataset_name=MindBridge
gpu_num=2
exo_input_len=52
num_meta=52
num_exo_vars=9


for seed in 21 42 63 84 105
    do  
        python -u run.py \
        --seed $seed \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --gpu_num $gpu_num \
        --exo_input_len $exo_input_len \
        --num_meta $num_meta \
        --num_exo_vars $num_exo_vars
    done