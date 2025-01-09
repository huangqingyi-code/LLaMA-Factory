MASTER_ADDR=10.205.23.133
MASTER_PORT=29500

NODE_RANK=0
NUM_GPUS_PER_NODE=8

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export NCCL_DEBUG="WARN"  # INFO 或 "WARN" 或 "ERROR" 以减少输出量
export NCCL_SOCKET_IFNAME="ens10f0np0"  # 替换为你的网络接口名
export NCCL_P2P_LEVEL="NVL"  # NVLINK

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train.py \
    --stage sft \
    --model_name_or_path /data3/models/Qwen/Qwen2-72B-Instruct \
    --template qwen \
    --do_train \
    --dataset all_data \
    --finetuning_type full \
    --output_dir /data3/sft_output/qwen2-instruct133-0910 \
    --overwrite_cache \
    --cache_dir /data3/sft_output/.cache_dir133-0910 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 32 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --save_strategy "steps" \
    --save_steps 200 \
    --eval_strategy "steps" \
    --val_size 0.02 \
    --eval_steps 100 \
    --cutoff_len 8192 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16 true \
    --deepspeed train_bash/deepspeed3.json \
    --report_to "none" \
    --flash_attn "fa2" \
    --ddp_timeout 36000