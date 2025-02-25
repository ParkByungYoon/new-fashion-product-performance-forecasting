model_name=TimeXer
gpu_num=1
num_endo_vars=4
num_exo_vars=48
num_layers=4

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_endo_vars $num_endo_vars \
    --num_exo_vars $num_exo_vars \
    --num_layers $num_layers