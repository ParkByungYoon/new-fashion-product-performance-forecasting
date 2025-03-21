model_name=Timer
gpu_num=1
input_len=52
num_vars=3
num_meta=96
segment_len=1
data_dir=/SSL_NAS/SFLAB/visuelle2
dataset_name=Visuelle
log_dir=log/Visuelle

python -u run.py \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --data_dir $data_dir \
    --log_dir $log_dir \
    --gpu_num $gpu_num \
    --input_len $input_len \
    --num_vars $num_vars \
    --num_meta $num_meta \