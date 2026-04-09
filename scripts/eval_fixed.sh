cd "$(dirname "$0")/.."
root_dir=/home/tjn2004/uav/UAV_ON
echo $PWD

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_2.py \
    --maxActions 150 \
    --eval_save_path $root_dir/logs/scene \
    --dataset_path /home/tjn2004/uav/DATASET/UAV-ON-data/valset/Slum.json \
    --is_fixed  true\
    --gpu_id 0 \
    --batchSize 1 \
    --simulator_tool_port 30000