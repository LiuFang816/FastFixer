"""
@Date: 2023.10.18
@Description:
    Extract test points from the original data(/data/data_structure_2023/raw/homeworks_all.json).
@Note:
    This script should be run after filter_data.py.
    THis script should only be run once.
"""

import json
from config import *
from xml.etree import ElementTree as ET
import os
import re
import html


EXTRACT_DATA = os.path.join(DATA_DIR, 'data_structure_2023')

problem_set = os.listdir(FILTERED_DIR)
problem_properties = {}

def extract_test_cases():
    with open(os.path.join(EXTRACT_DATA, 'raw', 'homeworks_all.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)['data']['homework_question']
    f.close()
    for homework in data:
        homework_id = homework['id']
        if not homework['ude']['type'] == 'program' and not homework['ude']['type'] == 'program-fill-gap':
            continue
        if 'problemID_' + str(homework_id) not in problem_set:
            continue
        description = homework['description']
        '''使用正则表达式去除description中的html格式'''
        description = re.sub(r'<[^>]+>', '', description)
        description = html.unescape(description)
        pattern = r"(【[^】]*】)"  
        lines = re.findall(pattern, description)
        things = []
        for line in lines:
            things.append(description.split(line)[0])
            if len(description.split(line)) > 1:
                description = description.split(line)[1]
        things = things[1:]
        real = dict(zip(lines[:-1], things))
        problem_properties['problemID_' + str(homework_id)] = real
    with open(os.path.join(RES_DIR, 'problem_description_extra_cn.json'), 'w', encoding='utf-8') as f:
        json.dump(problem_properties, f, ensure_ascii=False)
    f.close()


if __name__ == '__main__':
    extract_test_cases()