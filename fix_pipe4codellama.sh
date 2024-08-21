#!/bin/bash
cd `dirname $0`

BASE_DIR='/home/FastFixer/FastFixer/'
pyenv activate FastFixer
PYTHONPATH=$BASE_DIR python fix_pipe.py \
--model codellama-instrcut-7b-hf-huggingface \
--msg-passed buggy-reason \
--finetune-method lora_8k_length_4000_knowledge_mask_mark_random_between_0_and_0_8_take_2 \
--augment

# PYTHONPATH=$BASE_DIR python fix_pipe.py \
# --model codellama-instruct-7b-hf-huggingface \
# --msg-passed buggy-reason \
# --augment true

# PYTHONPATH=$BASE_DIR python fix_pipe.py \
# --model codellama-instrcut-7b-hf-huggingface \
# --msg-passed buggy-reason \
# --finetune-method lora_8k_length_4000_knowledge_mask_mark_random_between_0_and_0_8 \
# --no-augment