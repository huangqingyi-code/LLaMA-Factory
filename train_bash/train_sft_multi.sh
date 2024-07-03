MASTER_ADDR=10.205.23.131
MASTER_PORT=29500

NODE_RANK=0
NUM_GPUS_PER_NODE=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train.py \
    --stage sft \
    --model_name_or_path /data0/pretrained-models/Qwen2-7B-Instruct \
    --template qwen \
    --do_train \
    --dataset all_data \
    --finetuning_type full \
    --output_dir /data4/sft_output/qwen2-instruct131-0628 \
    --overwrite_cache \
    --cache_dir /data4/sft_output/.cache_dir131-0628 \
    --overwrite_output_dir \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 32 \
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
    --num_train_epochs 2 \
    --plot_loss \
    --bf16 true \
    --deepspeed train_bash/deepspeed2_multi.json \
    --report_to "wandb" \
    --flash_attn "fa2" \
    --ddp_timeout 3600 \
    &> logs/training1.log