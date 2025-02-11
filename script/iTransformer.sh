model_name=iTransformer
gpu_num=1
num_vars=43

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_vars $num_vars