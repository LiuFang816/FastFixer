import os
import shutil
from config import *

"""
@Date: 2023.09.09
@Description:
    Filter data to remove those students who do not have both correct and wrong answers.
    The filtered data will be stored in data/filtered.
    data/tmp is the data before filtering.(will be removed in the future)
    This script should be run after data_processing/extract_data.py and use only once.
"""


def is_path_avialable(path: str):
    try:
        wrong_cnt = len(os.listdir(os.path.join(path, 'wrong')))
        correct_cnt = len(os.listdir(os.path.join(path, 'correct')))
    except:
        return False
    if wrong_cnt == 0 or correct_cnt == 0:
        return False
    return True


def filter_data():
    for problem in os.listdir(os.path.join(EXTRACT_DIR)):
        for student in os.listdir(os.path.join(EXTRACT_DIR, problem)):
            if not is_path_avialable(os.path.join(EXTRACT_DIR, problem, student)):
                continue
            else:
                shutil.copytree(os.path.join(EXTRACT_DIR, problem, student),
                                os.path.join(FILTERED_DIR, problem, student))
                
if __name__ == '__main__':
    filter_data()