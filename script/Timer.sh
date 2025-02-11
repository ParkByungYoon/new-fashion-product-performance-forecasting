model_name=Timer
gpu_num=2
num_vars=43
segment_len=4

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_vars $num_vars \
    --segment_len $segment_len \