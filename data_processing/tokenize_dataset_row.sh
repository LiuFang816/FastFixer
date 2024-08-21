#!/bin/bash
cd `dirname $0`/..

# activate python env
source ./venv/bin/activate

# tokenize dataset row

BASE_DIR='/home/FastFixer/LMBPF'
DATASET='dataset'    

PYTHONPATH=$BASE_DIR python data_processing/tokenize_dataset_rows.py \
--jsonl_path ${BASE_DIR}/data/finetune_data/${DATASET}.jsonl \
--save_path ${BASE_DIR}/data/tokenized_data \
--skip_overlength True \
--max_seq_length 1500

