CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port=29503 src/evaluate.py \
    --model_name_or_path /data4/sft_output/qwen2-instruct132-0701/checkpoint-3200 \
    --template qwen \
    --task_dir /home/qyhuang/LLaMA-Factory/evaluation \
    --task cmmlu \
    --split test \
    --save_dir /home/qyhuang/LLaMA-Factory/output/cmmlu_5_qwen2_sft \
    --lang zh \
    --n_shot 5 \
    --batch_size 10