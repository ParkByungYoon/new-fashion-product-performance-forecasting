model_name=Timer
dataset_name=MindBridge
gpu_num=1
exo_input_len=52
num_exo_vars=9
num_meta=52
segment_len=4


python -u run.py \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --data_dir $data_dir \
    --log_dir $log_dir \
    --gpu_num $gpu_num \
    --input_len $input_len \
    --num_vars $num_vars \
    --num_meta $num_meta \