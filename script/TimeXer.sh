model_name=TimeXer
gpu_num=2
num_vars=45
num_endo_vars=3
num_layers=2

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_vars $num_vars \
    --num_endo_vars $num_endo_vars \
    --num_layers $num_layers