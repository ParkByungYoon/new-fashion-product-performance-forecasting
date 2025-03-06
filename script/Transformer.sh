model_name=Transformer
gpu_num=2
num_vars=45

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_vars $num_vars