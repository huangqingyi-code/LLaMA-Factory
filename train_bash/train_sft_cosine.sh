date=0814

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=29500 src/train.py \
    --stage sft \
    --model_name_or_path /data0/pretrained-models/Qwen2-7B-Instruct \
    --template qwen \
    --do_train \
    --dataset all_data \
    --finetuning_type full \
    --output_dir /data4/sft_output/qwen2-instruct-${date} \
    --overwrite_cache \
    --cache_dir /data4/sft_output/.cache_dir132 \
    --overwrite_output_dir \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 32 \
    --preprocessing_num_workers 32 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate":0.2}' \
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
    --deepspeed train_bash/deepspeed2.json \
    --report_to "wandb" \
    --flash_attn "fa2" \
    --ddp_timeout 3600 \
    &> logs/training-${date}.log