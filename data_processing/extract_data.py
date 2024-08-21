import os
import json
import shutil
from config import *


"""
@Date: 2023.09.09
@Description: Extract data from original data.
    Reformat the data to the following structure:
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
    │   │   └── ...
    │   ├── problemID_2
    │   ├── ...
@Note:
    The original data is stored in data/data_structure_2023(2022).
    This script should be run only once.
"""

REFORMET_DATA = 'data_structure_2022'

with open(os.path.join(DATA_DIR, REFORMET_DATA, 'gen', 'homework_questions_data.json')) as f:
    homework_question = json.load(f)
f.close()

for homework in homework_question:
    for problem in homework_question[homework]:
        for student in homework_question[homework][problem]:
            if not os.path.exists(os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student)):
                os.makedirs(os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student))
            correct_cnt = 0
            wrong_cnt = 0
            for submit_log in homework_question[homework][problem][student]:
                if not os.path.exists(os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student, 'correct')):
                    os.makedirs(os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student, 'correct'))
                if not os.path.exists(os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student, 'wrong')):
                    os.makedirs(os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student, 'wrong'))
                if 'attachmentInfo' not in submit_log:
                    continue
                attachment_info = submit_log['attachmentInfo']
                score = int(float(submit_log['score']['raw']))
                if len(attachment_info) > 1:
                    print(homework, problem, student)
                if len(attachment_info) == 0:
                    continue
                if score == 100:
                    try:
                        shutil.copy(os.path.join(DATA_DIR, REFORMET_DATA, 'gen', 'src', attachment_info[0]['sha2'] + '.c'),
                                os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student, 'correct', 'studentID_' + student + '_' + str(correct_cnt) + '.c'))
                    except:
                        continue
                    correct_cnt += 1
                else:
                    try:
                        shutil.copy(os.path.join(DATA_DIR, REFORMET_DATA, 'gen', 'src', attachment_info[0]['sha2'] + '.c'),
                                os.path.join(EXTRACT_DIR, 'problemID_' + problem, 'studentID_' + student, 'wrong', 'studentID_' + student + '_' + str(wrong_cnt) + '_pass_' + str(score / 20) + '.c'))
                    except:
                        continue
                    wrong_cnt += 1