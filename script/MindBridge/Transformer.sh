model_name=Transformer
dataset_name=MindBridge
gpu_num=1
exo_input_len=52
output_dim=1024
num_meta=52
num_exo_vars=2
num_epochs=20

for seed in 21 42 63 84 105
    do  
        python -u run.py \
        --seed $seed \
        --output_dim $output_dim \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --gpu_num $gpu_num \
        --exo_input_len $exo_input_len \
        --num_meta $num_meta \
        --num_exo_vars $num_exo_vars \
        --num_epochs $num_epochs\
        --use_endo \
        --use_revin
    done