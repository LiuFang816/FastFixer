import torch

torch.cuda.empty_cache()
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainer,
    AutoModelForCausalLM,
    Trainer,
)
from finetune.data_collator import DataCollatorForSeq2SeqKnowledgeMaskEnabled
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from finetune.llama_forward import forward_with_knowledge_mask
from optimum.bettertransformer import BetterTransformer
from peft import (
    LoraConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)
import datasets
import json
import os

DEBUG = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"
from config import *
from utils.get_problem_properties import *
from finetune.knowledge_mask import (
    get_line_level_knowledge_mask,
    get_input_id_level_knowledge_mask,
)

LlamaForCausalLM.forward = forward_with_knowledge_mask


data = datasets.load_dataset(
    "json", data_files=os.path.join(FINETUNE_DATA_DIR, "filtered.json"), split="train"
)
print(data)
print(type(data))

if not DEBUG:
    train_data = data.train_test_split(test_size=0.1)["train"]
    eval_data = data.train_test_split(test_size=0.1)["test"]
else:
    train_data = data.train_test_split(test_size=0.99)["train"].select(range(100))
    eval_data = data.train_test_split(test_size=0.01)["test"].select(range(100))
with open(TRAIN_BUGGY_LABEL_FILE, "r") as f:
    log_dict = json.load(f)
f.close()

local_rank = int(os.getenv("LOCAL_RANK", "0"))


training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=12000,
    learning_rate=3e-4,
    remove_unused_columns=False,
    bf16=True,
    logging_steps=100,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    gradient_checkpointing=True,
    eval_steps=10000,
    # remove_unused_columns=False,
    deepspeed={
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "offload_param": {"device": "cpu", "pin_memory": False},
            "" "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 3000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    },
    output_dir=FINETUNE_SHARED.FINETUNE_MODEL_PATH,
    save_total_limit=20,
    group_by_length=True,
)

base_model = "codellama/CodeLlama-7b-Instruct-hf"
if DEBUG:
    model = None
else:
    model = AutoModelForCausalLM.from_pretrained(base_model, use_flash_attention_2=True)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(prompt, line_level_knowledge_mask):
    result = tokenizer(
        # TODO: set max_length to 4000
        prompt,
        return_tensors=None,
        padding=False,
        max_length=4000,
        truncation=True,
    )
    result["labels"] = result["input_ids"].copy()
    result["knowledge_mask"] = get_input_id_level_knowledge_mask(
        result["input_ids"], line_level_knowledge_mask
    )
    return result


def extract_buggy_reason(wrong_file, log_dict):
    log_to_reason = {
        "Other Error": None,
        "Compile Error": "This code has compile error",
        "Presentation Error": "This code has presentation error, which means the output misses some spaces or newlines or has extra spaces or newlines",
        "Possible Presentation Error": "This code has presentation error, which means the output misses some spaces or newlines or has extra spaces or newlines",
        "Useless Debug Log": "This code has extra printf or puts which student use it for debug and forget to delete it or this code has extra characters in printf or puts",
        "Time Limit Exceeded": "This code has time limit exceeded error, it is possible that this code has deed loop",
        "Correct": "This code is correct, you do not need to fix it",
    }
    log = log_dict[wrong_file]
    reason = log_to_reason[log]
    return reason


def generate_and_tokenize_prompt(data_point):
    buggy_reason = extract_buggy_reason(data_point["wrong_file_path"], log_dict)
    problem_describe = get_problem_description(data_point["problem"])
    # input_format = get_input_format(data_point["problem"])
    # output_format = get_output_format(data_point["problem"])
    # testpoint_example = get_testpoint_example(data_point["problem"])
    wrong_code = data_point["wrong_code"]
    correct_code = f"""
[FIXED]
{data_point["correct_code"]}
[/FIXED]
"""
    buggy_reason_part = f"\n[BUGGY]\n{buggy_reason}\n[/BUGGY]" if buggy_reason else ""
    buggy_reason_desc = (
        "\nThe possible buggy reason starts with a [BUGGY] tag and ends with a [/BUGGY] tag."
        if buggy_reason
        else ""
    )
    prefix_prompt = f"""[INST]
You are an expert C programmer and teacher. Your student are doing program homework.
The problem description starts with a [P] tag and ends with a [/P] tag.
Your student's code starts with a [C] tag and end with a [/C] tag is buggy, you should fix his code.{buggy_reason_desc}
The fixed code should start with a [FIXED] tag and end with a [/FIXED] tag.
[/INST]
[P]
{problem_describe}
[/P]
[C]
{wrong_code}
[/C]{buggy_reason_part}
"""
    line_level_knowledge_mask = get_line_level_knowledge_mask(
        wrong_code,
        correct_code,
        prefix_prompt,
        delete_context_mask_len=0,
        # random_mask=True,
        random_mask=False,
        random_mask_ratio=0.8,
        all_context_mask_len=0,
        post_context_only=True,
    )
    prompt = prefix_prompt + correct_code
    return tokenize(prompt, line_level_knowledge_mask)


tokenized_train_data = train_data.map(generate_and_tokenize_prompt)
tokenized_eval_data = eval_data.map(generate_and_tokenize_prompt)

reversed_columns = ["attention_mask", "input_ids", "labels", "knowledge_mask"]
unused_columns = list(set(tokenized_train_data.column_names) - set(reversed_columns))
tokenized_train_data = tokenized_train_data.remove_columns(unused_columns)
tokenized_eval_data = tokenized_eval_data.remove_columns(unused_columns)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_alpha=16,
    target_modules=[
        "o_proj",
        "k_proj",
        "v_proj",
        "q_proj",
    ],
    r=8,
    lora_dropout=0.1,
)

if DEBUG:
    exit()
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
# model = BetterTransformer.transform(model)
model.train()


if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True


trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_eval_data,
    args=training_args,
    data_collator=DataCollatorForSeq2SeqKnowledgeMaskEnabled(tokenizer, return_tensors="pt", padding=True),
)

# model = torch.compile(model)
trainer.train()
