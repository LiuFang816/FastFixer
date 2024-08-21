#!/bin/bash
cd `dirname $0`/..

# activate python env
# source ./venv/bin/activate
# pyenv activate FastFixer

BASE_DIR='/home/FastFixer/FastFixer'

# output_dir will be replaced in finetune_chatglm.py
PYTHONPATH=$BASE_DIR torchrun --nproc_per_node=2 finetune/finetune_chatglm.py \
--lora_rank 8 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--max_steps 40000 \
--save_steps 10000 \
--save_total_limit 2 \
--learning_rate 1e-4 \
--fp16 \
--remove_unused_columns false \
--logging_steps 50 \
--output_dir $BASE_DIR/checkpoints