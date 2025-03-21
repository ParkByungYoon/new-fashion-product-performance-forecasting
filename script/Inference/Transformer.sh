model_name=Transformer
dataset_name=MindBridge
ckpt_name=Transformer-250312-1534.ckpt
gpu_num=3
exo_input_len=52
num_meta=52
num_exo_vars=3


python -u inference.py \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --ckpt_name $ckpt_name \
    --gpu_num $gpu_num \
    --exo_input_len $exo_input_len \
    --num_meta $num_meta \
    --num_exo_vars $num_exo_vars \
    --use_trend \
    # --use_meta_sale \
    # --use_weather \