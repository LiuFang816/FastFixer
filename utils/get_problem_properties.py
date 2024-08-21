import json
import os
from config import *

'''必须是全局，防止多次读取文件'''
with open(os.path.join(RES_DIR, 'problem_properties_extra.json'), 'r', encoding='utf-8') as f:
    problem_properties_extra = json.load(f)
f.close()

def get_problem_description(problem):
    if not '【问题描述】' in problem_properties_extra[problem]:
        return None
    return problem_properties_extra[problem]['【问题描述】']

def get_testpoint_example(problem):
    testpoint_example = {}
    with open(os.path.join(FILTERED_DIR, problem, 'testcase', '1.in'), 'r', encoding='utf-8') as f:
        testpoint_example['input'] = f.read()
    f.close()
    with open(os.path.join(FILTERED_DIR, problem, 'testcase', '1.out'), 'r', encoding='utf-8') as f:
        testpoint_example['output'] = f.read()
    f.close()
    return testpoint_example


def get_input_format(problem):
    if not '【输入形式】' in problem_properties_extra[problem]:
        return None
    return problem_properties_extra[problem]['【输入形式】']


def get_output_format(problem):
    if not '【输出形式】' in problem_properties_extra[problem]:
        return None
    return problem_properties_extra[problem]['【输出形式】']

def file_free_problem(problem_id):
    if problem_properties[problem_id]['is_input_file'] or problem_properties[problem_id]['is_output_file']:
        return False
    if problem_properties[problem_id]['has_extra_input']:
        return False
    return True