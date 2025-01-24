model_name=Fullformer
gpu_num=0
num_vars=3
segment_len=4

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_vars $num_vars \
    --segment_len $segment_len \