export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export NCCL_DEBUG="WARN"  # INFO 或 "WARN" 或 "ERROR" 以减少输出量
export NCCL_SOCKET_IFNAME="ens10f0np0"  # 替换为你的网络接口名
export NCCL_P2P_LEVEL="NVL"  # NVLINK

date=1018

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=29500 src/train.py \
    --stage sft \
    --model_name_or_path /data4/models/Qwen/Qwen2.5-7B-Instruct \
    --template qwen \
    --do_train \
    --dataset all_data \
    --finetuning_type full \
    --output_dir /data4/sft_output/qwen2.5-7b-ins-${date} \
    --overwrite_cache \
    --cache_dir /data4/sft_output/.cache_dir132 \
    --overwrite_output_dir \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 32 \
    --preprocessing_num_workers 64 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --save_strategy "steps" \
    --save_steps 300 \
    --eval_strategy "steps" \
    --val_size 0.02 \
    --eval_steps 300 \
    --cutoff_len 7800 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16 true \
    --deepspeed train_bash/deepspeed2_multi.json \
    --report_to "wandb" \
    --flash_attn "fa2" \
    --ddp_timeout 36000 \
    &> logs/training-${date}.log