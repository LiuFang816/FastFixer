# FastFixer: An Efficient and Effective Approach for Repairing Programming Assignments

## Introduction


  Providing personalized and timely feedback for programming assignments is useful for programming education. Automated program repair (APR) techniques have been used to fix the bugs in programming assignments, where the Large Language Models (LLMs) based approaches have shown promising results. Given the growing complexity of identifying and fixing errors in advanced programming assignments, current fine-tuning strategies for APR are inadequate in guiding the LLM to identify errors and make accurate edits during the generative repair process. Furthermore, the autoregressive decoding approach employed by the LLM could potentially impede the efficiency of the repair, thereby hindering the ability to provide timely feedback. To tackle these challenges, we propose FastFixer, an efficient and effective approach for programming assignment repair. To assist the LLM in accurately identifying and repairing bugs, we first propose a novel repair-oriented fine-tuning strategy, aiming to enhance the LLM's attention towards learning how to generate the necessary patch and its associated context. Furthermore, to speed up the patch generation, we propose an inference acceleration approach that is specifically tailored for the program repair task. The evaluation results demonstrate that FastFixer obtains an overall improvement of 20.46\% in assignment fixing when compared to the state-of-the-art baseline. Considering the repair efficiency, FastFixer achieves a remarkable inference speedup of $16.67\times$ compared to the autoregressive decoding algorithm.  

## Setup

### Pre-requisites
- gcc $\ge$ 9.30
- python $\ge$ 3.8
- cuda 11.8
- python $\ge$ 3.9.6

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
- Download the dataset from [here](https://figshare.com/s/a502270bf0d36774c791)
- Extract the dataset for finetune after filter to the `filtered` directory
- Extract the dataset for evaluation to the `dataset` directory

run the following command to tokenize the dataset, get a `jsonl` file for finetuning the model
```bash
bash data_processing/tokenizer_dataset_row.sh
```

### Model Finetuning

- You can change the crucial properties in the `config.py` file, such as backbone model, finetune method, etc.
- You can change other hyperparameters in the `finetune_codellama` file, such as learning rate, batch size, deepspeed hypermeters, etc.

run the following command to finetune the model, you don't need a `ds_config.json` file, for the deepspeed hypermeters is already set in the `finetune_codellama` file.
```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false deepspeed --num_gpus=[num of gpus you want to use] finetune/finetune_codellama.py --deepspeed ds_config.json
```

### Fix and Evaluate

- You can change the crucial properties in the `config.py` file, such as backbone model, finetune method, etc.

run the following command to fix the dataset. You can set weather to augment the model by adding `--augment` or `--no-augment` parameters in the `fix_pipe4codellama.sh` file. 
```bash
PYTHONPATH=. bash fix_pipe4codellama.sh
```

run the following command to evaluate the buggy files and statistic buggy types.
```bash
PYTHONPATH=. bash buggy_related.sh
```

### Gudiance 

the Core implementation of FastFixer is in the `finetune/knowledge_mask.py` file and `fix/llama_decoding` directory. The `finetune/knowledge_mask.py` file is used to generate the modification-focused mask for the model, and the `fix/llama_decoding` directory is used accelerate the inference process.
