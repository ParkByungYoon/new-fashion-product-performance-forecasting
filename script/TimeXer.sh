model_name=TimeXer
gpu_num=0
num_endo_vars=3
num_exo_vars=47

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_endo_vars $num_endo_vars \
    --num_exo_vars $num_exo_vars \