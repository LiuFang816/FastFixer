import os
import re
import json
import jsonlines
from config import *
from multiprocessing import Pool, Manager
from utils.calcs import edit_distance
from utils.get_problem_properties import *


def clear_annotation(code: str) -> str:
    """
    Remove the annotation in the binary code.
    """
    pattern = b"/*"
    in_comment = False
    result = bytearray()
    i = 0
    while i < len(code):
        if not in_comment and code[i:i+2] == pattern:
            in_comment = True
            i += 2
        elif in_comment and code[i:i+2] == b"*/":
            in_comment = False
            i += 2
        elif not in_comment and code[i:i+2] == b"//":
            # Skip to end of line
            while i < len(code) and code[i] != ord('\n'):
                i += 1
            i += 1
        elif not in_comment:
            result.append(code[i])
            i += 1
        else:
            i += 1
    return bytes(result)



def reformat_data(wrong_file, correct_file):
    """
    Reformat the data into finetune_data.
    """
    with open(wrong_file, "rb") as f:
        wrong_code = f.read()
    with open(correct_file, "rb") as f:
        correct_code = f.read()
    wrong_code = clear_annotation(wrong_code)
    correct_code = clear_annotation(correct_code)
    '''someone used both utf-8 and gbk, cannot deal with it now, return None at this situation'''
    try:
        wrong_code = wrong_code.decode("utf-8")
    except UnicodeDecodeError:
        try:
            wrong_code = wrong_code.decode("gbk")
        except:
            return None
    try:
        correct_code = correct_code.decode("utf-8")
    except UnicodeDecodeError:
        try:
            correct_code = correct_code.decode("gbk")
        except:
            return None
    return wrong_code, correct_code
    


"""
this is for filter the data
@:param wrong_code: the wrong code
@:param correct_code: the correct code
@:return: True or False

princples:
1. the edit distance should be less than 10
TODO: maybe add more filter principles
"""
def is_data_avaiable(wrong_code, correct_code):
    # calculate the edit distance
    distance_between_code = edit_distance(wrong_code, correct_code)
    if distance_between_code > 10:
        return False
    return True


def reformat_for_student(student_path, res): 
    for correct_file in os.listdir(os.path.join(student_path, 'correct')):
        if not correct_file.endswith('.c'):
            continue
        for wrong_file in os.listdir(os.path.join(student_path, 'wrong')):
            wrong_file = os.path.join(student_path, 'wrong', wrong_file)
            if not wrong_file.endswith('.c'):
                continue
            correct_file = os.path.join(student_path, 'correct', correct_file)
            reformated_data = reformat_data(wrong_file, correct_file)
            if reformated_data is None:
                continue
            wrong_code = reformated_data[0]
            correct_code = reformated_data[1]
            if not is_data_avaiable(wrong_code, correct_code):
                continue
            # res.append({
            #     "context": "Instruction: Fix bugs for the fllowing code, {}".format(
            #         wrong_code
            #     ),
            #     "target": correct_code,
            #     "wrong_file_path": wrong_file,
            #     "correct_file_path": correct_file
            # })
            res.append({
                "wrong_code": wrong_code,
                "correct_code": correct_code,
                "wrong_file_path": wrong_file,
                "correct_file_path": correct_file,
                "problem": student_path.split('/')[-2],
            })

def reformat():
    """
    Main function.
    """
    res = Manager().list()
    pl = Pool(CPU_COUNT)

    for problem in os.listdir(os.path.join(TRAIN_DATA_DIR)):
        problem_id = problem.replace("problemID_", "")
        if not file_free_problem(problem_id):
            continue
        problem_path = os.path.join(TRAIN_DATA_DIR, problem)
        for student in os.listdir(problem_path):
            if not student.startswith('student'):
                continue
            student_path = os.path.join(problem_path, student)
            pl.apply_async(reformat_for_student, args=(student_path, res))

    pl.close()
    pl.join()
    
    # convert res to jsonlines file
    with jsonlines.open(os.path.join(FINETUNE_DATA_DIR, TRAIN_DATASET + '.jsonl'), "w") as f:
        f.write_all(list(res))
    f.close()

    # convert res to json file
    with open(os.path.join(FINETUNE_DATA_DIR, TRAIN_DATASET + '.json'), "w") as f:
        json.dump(list(res), f, indent=4)
    f.close()


if __name__ == "__main__":
    reformat()