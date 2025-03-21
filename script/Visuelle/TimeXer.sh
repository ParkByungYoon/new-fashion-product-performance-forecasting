model_name=TimeXer
dataset_name=Visuelle
gpu_num=3
endo_input_len=52
exo_input_len=52
segment_len=4
num_meta=96
num_endo_vars=1
num_exo_vars=3


for seed in 21 42 63 84 105
    do  
        python -u run.py \
        --seed $seed \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --gpu_num $gpu_num \
        --endo_input_len $endo_input_len \
        --exo_input_len $exo_input_len \
        --segment_len $segment_len \
        --num_meta $num_meta \
        --num_endo_vars $num_endo_vars \
        --num_exo_vars $num_exo_vars
    done