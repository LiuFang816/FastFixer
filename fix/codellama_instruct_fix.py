# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
from config import *
from utils.extract_code import extract_code_in_markdown_format, extract_code_embrace_by_label
from utils.file_reader import read_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from peft import PeftModel
from utils.get_problem_properties import *
import time

'''经测试, MAX_LENGTH在5000时, 最高占用大概17GB显存/卡,共2卡'''
MAX_LENGTH = 5000


def fix_by_huggingface_version_codellama(augment, wrong_code: str, generator, tokenizer, problem_describe, testpoint_example, buggy_reason, input_format, output_format):
    if buggy_reason is not None:
        PROMPT = f'''[INST]
You are an expert C programmer and teacher. Your student are doing program homework.
The problem description starts with a [P] tag and ends with a [/P] tag.
The input format description starts with a [I] tag and ends with a [/I] tag.
The output format description starts with a [O] tag and ends with a [/O] tag.
The testpoint example starts with a [T] tag and ends with a [/T] tag.
Your student's code starts with a [C] tag and end with a [/C] tag is buggy, you should fix his code.
The possible buggy reason starts with a [BUGGY] tag and ends with a [/BUGGY] tag.
The fixed code should start with a [FIXED] tag and end with a [/FIXED] tag.
[/INST]
[P]
{problem_describe}
[/P]
[I]
{input_format}
[/I]
[O]
{output_format}
[/O]
[T]
input:
{testpoint_example['input']}
output:
{testpoint_example['output']}
[/T]
[C]
{wrong_code}
[/C]
[BUGGY]
{buggy_reason}
[/BUGGY]
'''
    else:
        PROMPT = f'''[INST]
You are an expert C programmer and teacher. Your student are doing program homework.
The problem description starts with a [P] tag and ends with a [/P] tag.
The input format description starts with a [I] tag and ends with a [/I] tag.
The output format description starts with a [O] tag and ends with a [/O] tag.
The testpoint example starts with a [T] tag and ends with a [/T] tag.
Your student's code starts with a [C] tag and end with a [/C] tag is buggy, you should fix his code.
The fixed code should start with a [FIXED] tag and end with a [/FIXED] tag.
[/INST]
[P]
{problem_describe}
[/P]
[I]
{input_format}
[/I]
[O]
{output_format}
[/O]
[T]
input:
{testpoint_example['input']}
output:
{testpoint_example['output']}
[/T]
[C]
{wrong_code}
[/C]
'''
        
    PROMPT_WITH_WRONG_CODE = PROMPT + f'''
[FIXED]
{wrong_code}
[/FIXED]
'''

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].cuda()
    # input_ids = tokenizer(PROMPT, return_tensors="pt").cuda()
    if len(input_ids[0]) > MAX_LENGTH:
        return None
    prefix_length = len(input_ids[0])
    
    # output = generator.generate(input_ids, max_length=MAX_LENGTH, do_sample=False, top_p=0.95, top_k=60, temperature=0.8, num_return_sequences=1)
    if augment:
        prompt_with_wrong_code_ids = tokenizer(PROMPT_WITH_WRONG_CODE, return_tensors="pt")["input_ids"].cuda(0)
        wrong_code_ids = prompt_with_wrong_code_ids[:, input_ids.shape[1]:]
        output, forward_cnt = generator.generate(input_ids=input_ids, wrong_code=wrong_code_ids, max_length=MAX_LENGTH, do_sample=False)
    else:
        output, forward_cnt = generator.generate(input_ids=input_ids, max_length=MAX_LENGTH, do_sample=False)
        # output = generator.generate(**input_ids, max_length=MAX_LENGTH, do_sample=False)
    generation = tokenizer.batch_decode(output[:, input_ids.shape[-1]:], skip_special_tokens = True)[0]
    generated_length = len(output[:, input_ids.shape[-1]:][0])
    code = extract_code_embrace_by_label(generation, 'FIXED')
    '''sometimes the model forget the asked format, so we need to try to extract code in markdown format'''
    if code is None:
        code = extract_code_in_markdown_format(generation)
    return code, forward_cnt, prefix_length, generated_length

def extract_buggy_reason(wrong_file, log_dict):
    log_to_reason = {
        "Other Error": None,
        "Compile Error": "This code has compile error",
        "Presentation Error": "This code has presentation error, which means the output misses some spaces or newlines or has extra spaces or newlines",
        "Possible Presentation Error": "This code has presentation error, which means the output misses some spaces or newlines or has extra spaces or newlines",
        "Useless Debug Log": "This code has extra printf or puts which student use it for debug and forget to delete it or this code has extra characters in printf or puts",
        "Time Limit Exceeded": "This code has time limit exceeded error, it is possible that this code has deed loop",
        "Correct": "This code is correct, you do not need to fix it"
    }
    log = log_dict[wrong_file]
    reason = log_to_reason[log]
    return reason

def codellama_instuct_fix(augment):
    generator = None
    tokenizer = None
    generator = AutoModelForCausalLM.from_pretrained(
        'codellama/CodeLlama-7b-Instruct-hf',
        torch_dtype=torch.float16,
        device_map='auto')
    if FINETUNE_SHARED.IS_FINETUNED:
        generator = PeftModel.from_pretrained(generator, FINETUNE_SHARED.FINETUNE_MODEL_PATH + '/checkpoint-8000')
    tokenizer = AutoTokenizer.from_pretrained(
        'codellama/CodeLlama-7b-Instruct-hf')
    
    import json
    with open(BUGGY_CASE_PASS_NUM_FILE, "r") as f:
        buggy_pass_num_dict = json.load(f)
    f.close()
    with open(TEST_BUGGY_LABEL_FILE, "r") as f:
        log_dict = json.load(f)
    f.close()

    for problem in os.listdir(FIX_DATA_DIR):
        problem_id = problem.replace("problemID_", "")
        problem_path = os.path.join(FIX_DATA_DIR, problem)
        for student in os.listdir(problem_path):
            if not student.startswith('student'):
                continue
            student_path = os.path.join(problem_path, student)
            # if os.path.exists(os.path.join(student_path, FINETUNE_SHARED.FIXED_DIR)):
            #     continue
            for wrong_file in os.listdir(os.path.join(student_path, 'wrong')):   
                if not wrong_file.endswith('.c'):
                    continue
                if not wrong_file in buggy_pass_num_dict[problem]:
                    continue
                wrong_file = os.path.join(student_path, 'wrong', wrong_file)
                try:
                    wrong_code = read_file(wrong_file)
                except:
                    continue
                reason = extract_buggy_reason(wrong_file, log_dict)
                start_time = time.time()
                code_fixed, forward_cnt, prefix_length, generated_length = fix_by_huggingface_version_codellama(augment, wrong_code, generator, tokenizer, get_problem_description(problem), get_testpoint_example(problem), reason, get_input_format(problem), get_output_format(problem))
                end_time = time.time()
                if code_fixed is None:
                    continue
                if not os.path.exists(os.path.join(student_path, FINETUNE_SHARED.FIXED_DIR)):
                    os.mkdir(os.path.join(student_path, FINETUNE_SHARED.FIXED_DIR))
                with open(os.path.join(student_path, FINETUNE_SHARED.FIXED_DIR, os.path.basename(wrong_file)), 'w') as f:
                    try:
                        f.write(code_fixed)
                    except:
                        pass
                f.close()


if __name__ == "__main__":
    codellama_instuct_fix()
