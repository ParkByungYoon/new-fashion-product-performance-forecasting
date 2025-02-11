model_name=Crossformer
gpu_num=1
num_vars=50
segment_len=4

python -u run.py \
    --model_name $model_name \
    --gpu_num $gpu_num \
    --num_vars $num_vars \
    --segment_len $segment_len \