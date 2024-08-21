"""
@Date: 2023.09.09
@Description:
    Extract test points from the original data(/data/data_structure_2023/raw/homeworks_all.json).
    testcase files will be stored in /data/data_structure_2023/filtered/problemID_xxx/testcase.
    the final data structure will be:
        data
    ├── filtered
    │   ├── problemID_1
    │   │   ├── studentID_1
    │   │   │   ├── correct
    │   │   │   │   ├── studentID_1_0.c
    │   │   │   │   ├── studentID_1_1.c
    │   │   │   │   └── ...
    │   │   │   └── wrong
    │   │   │       ├── studentID_1_0_pass_0.5.c
    │   │   │       ├── studentID_1_1_pass_0.5.c
    │   │   │       └── ...
    │   │   ├── studentID_2
    │   │   ├── ...
    │   │   └── testcase
    │   ├── problemID_2
    │   ├── ...
    and inside the testcase folder, there are three main types of files:
    1. *.in: input file for the test point.
    2. *.out: output file for the test point.
    3. *.extra: extra input file for the test point.
    and in config file(stdout of this script), there are some properties for each problem:
    1. is_output_file: whether the test case use file/stdout as output form.
    2. is_input_file: whether the test point use file/stdin as input form.
    3. has_extra_input: whether the test point has extra input file.
    4. input_file_name: the name of the input file(when is_input_file is True).
    5. output_file_name: the name of the output file(when is_output_file is True).
    6. extra_input_file_name: the name of the extra input file(when has_extra_input is True).
@Note:
    This script should be run after filter_data.py.
    THis script should only be run once.
"""

import json
from config import *
from xml.etree import ElementTree as ET
import os
import shutil

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
        is_output_file = True if homework['ude']['output'] != 'stdout' else False
        is_input_file = True if homework['ude']['input'] != 'stdin' else False
        has_extra_input = False
        input_file_name = ''
        output_file_name = ''
        extra_input_file_name = ''
        testdata = homework['ude']['testdata']
        testdata_xml = ET.fromstring(testdata)
        if 'problemID_' + str(homework_id) not in problem_set:
            continue
        test_cases_num = int(testdata_xml.attrib['count'])
        if not os.path.exists(os.path.join(FILTERED_DIR, 'problemID_' + homework_id, 'testcase')):
            os.mkdir(os.path.join(FILTERED_DIR, 'problemID_' + homework_id, 'testcase'))
        for i in range(int(test_cases_num)):
            test_case = testdata_xml.findall('testData' + str(i + 1))[0]
            if is_input_file:
                input_file_name = test_case.findall('input')[0].attrib['filename']
            test_point_input = test_case.findall('input')[0].text
            if len(test_case.findall('file')) > 0:
                has_extra_input = True
                extra_input = test_case.findall('file')[0].text
                extra_input_file_name = test_case.findall('file')[0].attrib['filename']
                with open(os.path.join(FILTERED_DIR, 'problemID_' + homework_id, 'testcase', str(i + 1) + '.extra'), 'w', encoding='utf-8') as f:
                    if extra_input is None:
                        extra_input = ''
                    f.write(extra_input)
                f.close()
            if is_output_file:
                output_file_name = test_case.findall('output')[0].attrib['filename']
            test_point_output = test_case.findall('output')[0].text
            with open(os.path.join(FILTERED_DIR, 'problemID_' + homework_id, 'testcase', str(i + 1) + '.in'), 'w', encoding='utf-8') as f:
                f.write(test_point_input)
            f.close()
            with open(os.path.join(FILTERED_DIR, 'problemID_' + homework_id, 'testcase', str(i + 1) + '.out'), 'w', encoding='utf-8') as f:
                if test_point_output is None:
                    test_point_output = ''
                f.write(test_point_output)
            f.close()
        problem_properties[homework_id] = {}
        problem_properties[homework_id]['is_output_file'] = is_output_file
        problem_properties[homework_id]['is_input_file'] = is_input_file
        problem_properties[homework_id]['has_extra_input'] = has_extra_input
        problem_properties[homework_id]['input_file_name'] = input_file_name
        problem_properties[homework_id]['output_file_name'] = output_file_name
        problem_properties[homework_id]['extra_input_file_name'] = extra_input_file_name
    print(problem_properties)


def copy_testcase():
    for problem in os.listdir(FIX_DATA_DIR):
        if problem in problem_set and not os.path.exists(os.path.join(FIX_DATA_DIR, problem, 'testcase')):
            shutil.copytree(os.path.join(FILTERED_DIR, problem, 'testcase'), os.path.join(FIX_DATA_DIR, problem, 'testcase'))

if __name__ == '__main__':
    extract_test_cases()
    copy_testcase()