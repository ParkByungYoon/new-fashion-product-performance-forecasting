model_name=Crossformer
dataset_name=Visuelle
gpu_num=1
exo_input_len=52
output_dim=256
segment_len=4
num_meta=96
num_exo_vars=5

for seed in 21 42 63 84 105
    do  
        python -u run.py \
        --seed $seed \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --gpu_num $gpu_num \
        --output_dim $output_dim \
        --exo_input_len $exo_input_len \
        --segment_len $segment_len \
        --num_meta $num_meta \
        --num_exo_vars $num_exo_vars \
        --use_endo \
        --use_revin
    done