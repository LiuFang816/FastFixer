"""
@Date: 2023.09.10
@Description: Evaluate the buggy code.
    Two Form:
    1. Train Data
        The data is in the following structure:
        data
        ├── filtered
        │   ├── problemID_1
        │   │   ├── studentID_1
        │   │   │   ├── buggy
        │   │   │   │   ├── studentID_1_0.c
        │   │   │   │   ├── studentID_1_1.c
        │   │   │   │   └── ...
        │   │   │   └── correct
        │   │   │       ├── studentID_1_0.c
        │   │   │       ├── studentID_1_1.c
        │   │   │       └── ...
        │   │   ├── studentID_2
        │   │   ├── ...
        │   │   └── testcase
        │   ├── problemID_2
        │   ├── ...

    2. Test Data
"""
import os
import json
import shutil
import subprocess
import re
from multiprocessing.pool import Pool
from multiprocessing import Manager
from config import *
from utils.get_problem_properties import *
import argparse


PRE_EXECUTE_NEEDED = True
TEST = True
LABEL = False
TEST_FIXED = True
EVALUATE_DIR = ''
TEST_SUBDIR = ''
LOG_FILE = ''

def get_global_value(test, pre_execute_needed, label, test_fixed):
    global TEST, PRE_EXECUTE_NEEDED, LABEL, TEST_FIXED, EVALUATE_DIR, TEST_SUBDIR, LOG_FILE
    TEST = test
    PRE_EXECUTE_NEEDED = pre_execute_needed
    LABEL = label
    TEST_FIXED = test_fixed
    EVALUATE_DIR = os.path.join(DATA_DIR, 'dataset') if TEST else os.path.join(DATA_DIR, 'filtered')
    if LABEL:
        if TEST:
            if TEST_FIXED:
                LOG_FILE = os.path.join(STATISTIC_RES_DIR, f"buggy_analyze_test_{FINETUNE_SHARED.FIXED_DIR}.json")
            else:
                LOG_FILE = os.path.join(STATISTIC_RES_DIR, "buggy_analyze_test.json")
        else:
            assert TEST_FIXED == False, "Train data can not be fixed."
            LOG_FILE = os.path.join(STATISTIC_RES_DIR, "buggy_analyze_train.json")
    else:
        assert TEST == True, "Train data should only be labeled"
        if TEST_FIXED:
            LOG_FILE = os.path.join(MODEL_FIX_RES_DIR, FINETUNE_SHARED.FIXED_DIR) + '.json'
        else:
            LOG_FILE = os.path.join(BUGGY_RES_DIR, "res.json")
    TEST_SUBDIR = 'wrong' if not TEST else (FINETUNE_SHARED.FIXED_DIR if TEST_FIXED else 'wrong')

"""
    @Description: Evaluate the buggy file for train data.
    @Args:
        problem: The problem id.
        student: The student id.
        file: The file name.
    @Returns:
        execute_res: The result of the execution.
        couple of situations:
        1. compile_error: The file can not be compiled.
        return {"compile_error": "Unknown Error/Error Message"}
        2. runtime_error: The file can be compiled, but it can not be executed.
        return {"input_file_id": {"runtime_error": "Unknown Error/Error Message"}}
        3. other_error: only when this problem requires output file, yet the output file is not generated.
        return {"input_file_id": {"other_error": "did not find output file"}}
        4. output: The output of the execution.
        return {"input_file_id": {"output": "output"}}
    """
def evalute_buggy_file_worker(problem, student, file):
    execute_res = {}
    problem_id = problem.replace("problemID_", "")
    problem_proerty = problem_properties[problem_id]
    if not compile(problem, student, file):
        execute_res["compile_error"] = "Unknown Error"
        return execute_res
    for testcase_file in os.listdir(os.path.join(EVALUATE_DIR, problem, 'testcase')):
        if not testcase_file.endswith(".in"):
            continue
        input_file_id = testcase_file.replace(".in", "")
        execute_res[input_file_id] = {}
        prepare_test_file(problem, input_file_id, problem_proerty)
        cmd = f"(cd {WORKSPACE} && timeout 2 ./{file}.out)"
        if not problem_proerty["is_input_file"]:
            cmd = f"(cd {WORKSPACE} && timeout 2 ./{file}.out < {os.path.join(EVALUATE_DIR, problem, 'testcase', input_file_id + '.in')})"
        try:
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 124:
                execute_res[input_file_id]["Time Limit Exceeded"] = ""
                continue
            if p.returncode != 0:
                execute_res[input_file_id]["runtime_error"] = p.stderr.decode("utf-8")
                continue
            if problem_proerty["is_output_file"]:
                output_file_name = problem_proerty["output_file_name"]
                try:
                    with open(os.path.join(WORKSPACE, output_file_name), "r") as f:
                        execute_res[input_file_id]["output"] = f.read()
                    f.close()
                except:
                    execute_res[input_file_id]["other_error"] = "did not find output file"
            else:
                execute_res[input_file_id]["output"] = p.stdout.decode("utf-8")
        except:
            execute_res[input_file_id]["runtime_error"] = "Unknown Error"
    return execute_res
        

"""
@Description: Compile the file.
@Returns:
    True: Compile successfully.
    False: Compile failed.
"""
def compile(problem, student, file):
    cmd = f"gcc -std=c99 -w {os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file)} -o {os.path.join(WORKSPACE, file)}.out"
    try:
        p = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        if p.returncode != 0:
            return False
        return True
    except:
        return False


def prepare_test_file(problem, input_file_id, problem_proerty):
    if problem_proerty["is_input_file"]:
        input_file_name = problem_proerty["input_file_name"]
        shutil.copyfile(
            os.path.join(EVALUATE_DIR, problem, "testcase", input_file_id + ".in"),
            os.path.join(WORKSPACE, input_file_name)
        )
    if problem_proerty["has_extra_input"]:
        extra_input_file_name = problem_proerty["extra_input_file_name"]
        shutil.copyfile(
            os.path.join(EVALUATE_DIR, problem, "testcase", input_file_id + ".extra"),
            os.path.join(WORKSPACE, extra_input_file_name)
        )
    if problem_proerty["is_output_file"]:
        output_file_name = problem_proerty["output_file_name"]
        if os.path.exists(os.path.join(WORKSPACE, output_file_name)):
            os.remove(os.path.join(WORKSPACE, output_file_name))


def check_presentation_error_for_case(output, answer):
    '''replace all the empty characters and new line characters'''
    output = re.sub(r"\s", "", output)
    answer = re.sub(r"\s", "", answer)
    return output == answer

def remove_trailing_whitespace_lines(input):
    output = input.rstrip()
    processed_lines = [line.rstrip() for line in output.splitlines()]
    return '\n'.join(processed_lines)


def check_correct_output_for_case(output, answer):
    return output.rstrip() == answer.rstrip()


def check_useless_debug_log_output_for_case(output, answer):
    '''check if answer is a subsequence of output'''
    output = re.sub(r"\s", "", output)
    answer = re.sub(r"\s", "", answer)
    if output == answer:
        return False
    i = 0
    for j in range(len(output)):
        if answer[i] == output[j]:
            i += 1
        if i == len(answer):
            return True
    return False


def check_presentaion_error_or_useless_debug_log_for_file(problem, execute_res):
    ea = 0
    ac = 0
    pe = 0
    ud = 0
    for testcase_file in os.listdir(os.path.join(EVALUATE_DIR, problem, 'testcase')):
        if not testcase_file.endswith(".out"):
            continue
        input_file_id = testcase_file.replace(".out", "")
        with open(os.path.join(EVALUATE_DIR, problem, 'testcase', testcase_file), "r") as f:
            answer = f.read()
        f.close()
        if not "output" in execute_res[input_file_id]:
            return False, False, False, False
        output = execute_res[input_file_id]["output"]
        answer = remove_trailing_whitespace_lines(answer)
        output = remove_trailing_whitespace_lines(output)
        if answer == "" and output == "":
            ea += 1
        elif check_correct_output_for_case(output, answer):
            ac += 1
        elif check_presentation_error_for_case(output, answer):
            pe += 1
        elif check_useless_debug_log_output_for_case(output, answer):
            ud += 1 
    return ac + ea == len(execute_res), pe + ea == len(execute_res), ud + ea == len(execute_res), ac + pe + ea == len(execute_res)


def get_pass_matrix(problem, execute_res):
    matrix = {}
    ac = True
    ac_num = 0
    for testcase_file in os.listdir(os.path.join(EVALUATE_DIR, problem, 'testcase')):
        if not testcase_file.endswith(".out"):
            continue
        input_file_id = testcase_file.replace(".out", "")
        with open(os.path.join(EVALUATE_DIR, problem, 'testcase', testcase_file), "r") as f:
            answer = f.read()
        f.close()
        if "runtime_error" in execute_res[input_file_id]:
            matrix[input_file_id] = 2
            ac = False
            continue
        if "Time Limit Exceeded" in execute_res[input_file_id]:
            matrix[input_file_id] = 4
            ac = False
            continue
        output = execute_res[input_file_id]["output"]
        answer = remove_trailing_whitespace_lines(answer)
        output = remove_trailing_whitespace_lines(output)
        if check_correct_output_for_case(output, answer):
            matrix[input_file_id] = 0
            ac_num += 1
        else:
            matrix[input_file_id] = 3
            ac = False
    return ac, ac_num, matrix



def execure_and_analyze_bug_type_worker(problem, student, file, buggy_analyze_dict, lock):
    if PRE_EXECUTE_NEEDED:
        execute_res = evalute_buggy_file_worker(problem, student, file)
        with open(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file.replace(".c", ".json")), "w") as f:
            f.write(json.dumps(execute_res))
        f.close()
    else:
        if os.stat(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file.replace(".c", ".json"))).st_size > 1024 * 1024 * 64:
            return
        with open(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file.replace(".c", ".json")), "r") as f:
            execute_res = json.load(f)
        f.close()
    file_absolut_path = os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file)
    with lock:
        if "compile_error" in execute_res:
            buggy_analyze_dict[file_absolut_path] = "Compile Error"
            return
        for input_file_id in execute_res:
            each_case_res = execute_res[input_file_id]
            if "Time Limit Exceeded" in each_case_res:
                buggy_analyze_dict[file_absolut_path] = "Time Limit Exceeded"
                return
        ac, pe, ud, possible_pe = check_presentaion_error_or_useless_debug_log_for_file(problem, execute_res)
        if ac:
            buggy_analyze_dict[file_absolut_path] = "Correct"
            return
        if pe:
            buggy_analyze_dict[file_absolut_path] = "Presentation Error"
            return
        if ud:
            buggy_analyze_dict[file_absolut_path] = "Useless Debug Log"
            return
        if possible_pe:
            buggy_analyze_dict[file_absolut_path] = "Possible Presentation Error"
            return
        buggy_analyze_dict[file_absolut_path] = "Other Error"
    
        
def evalute_and_label():
    pool = Pool(CPU_COUNT)
    with Manager() as manager:
        buggy_analyze_dict = manager.dict()
        lock = manager.Lock()
        for problem in os.listdir(os.path.join(EVALUATE_DIR)):
            if not problem.startswith("problemID_"):
                continue
            for student in os.listdir(os.path.join(EVALUATE_DIR, problem)):
                if not student.startswith("studentID_"):
                    continue
                if not os.path.exists(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR)):
                    continue
                for file in os.listdir(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR)):
                    if not file.endswith(".c"):
                        continue
                    pool.apply_async(execure_and_analyze_bug_type_worker, args=(problem, student, file, buggy_analyze_dict, lock))
        pool.close()
        pool.join()
        with open(LOG_FILE, "w") as f:
            json.dump(dict(buggy_analyze_dict), f)
        f.close()


def execure_only_worker(problem, student, file, fixed_analyze_dict, buggy_pass_num, lock):
    if PRE_EXECUTE_NEEDED:
        execute_res = evalute_buggy_file_worker(problem, student, file)
        with open(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file.replace(".c", ".json")), "w") as f:
            f.write(json.dumps(execute_res))
        f.close()
    else:
        with open(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file.replace(".c", ".json")), "r") as f:
            execute_res = json.load(f)
        f.close()
    file_absolut_path = os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR, file)
    with lock:
        if problem not in fixed_analyze_dict["detail"]:
            detail = fixed_analyze_dict["detail"]
            detail[problem] = {}
            fixed_analyze_dict["detail"] = detail
        if "compile_error" in execute_res:
            fixed_analyze_dict["total_compile_error_num"] += 1
            detail = fixed_analyze_dict["detail"]
            problem_dict = detail[problem]
            problem_dict[file_absolut_path] = 'compile_error'
            detail[problem] = problem_dict
            fixed_analyze_dict["detail"] = detail
            return
        ac, ac_num, matrix = get_pass_matrix(problem, execute_res)
        fixed_analyze_dict["case_pass_num"] += ac_num
        detail = fixed_analyze_dict["detail"]
        problem_dict = detail[problem]
        problem_dict[file_absolut_path] = matrix
        detail[problem] = problem_dict
        fixed_analyze_dict["detail"] = detail
        if TEST_FIXED:
            if ac:
                fixed_analyze_dict["total_pass_num"] += 1
            elif ac_num > buggy_pass_num:
                fixed_analyze_dict["pass_more_num"] += 1
            elif ac_num < buggy_pass_num:
                fixed_analyze_dict["pass_less_num"] += 1
            elif ac_num == buggy_pass_num:
                fixed_analyze_dict["pass_equal_num"] += 1


def evalute_only():
    pool = Pool(CPU_COUNT)
    with Manager() as manager:
        analyze_dict = manager.dict()
        lock = manager.Lock()
        if TEST_FIXED:
            with open(BUGGY_CASE_PASS_NUM_FILE, "r") as f:
                buggy_pass_num_dict = json.load(f)
            f.close()
        analyze_dict["pass_more_num"] = 0
        analyze_dict["pass_less_num"] = 0
        analyze_dict["pass_equal_num"] = 0
        analyze_dict["total_pass_num"] = 0
        analyze_dict["total_compile_error_num"] = 0
        analyze_dict["case_pass_num"] = 0
        analyze_dict["detail"] = {}
        for problem in os.listdir(os.path.join(EVALUATE_DIR)):
            if not problem.startswith("problemID_"):
                continue
            # if problem not in ['problemID_4559', 'problemID_4565', 'problemID_4937', 'problemID_4605']:
            #     continue
            problem_id = problem.replace("problemID_", "")
            if not file_free_problem(problem_id):
                continue
            for student in os.listdir(os.path.join(EVALUATE_DIR, problem)):
                if not student.startswith("studentID_"):
                    continue
                # if not os.path.exists(os.path.join(EVALUATE_DIR, problem, student, 'chatglmlora')):
                #     continue
                if not os.path.exists(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR)):
                    continue
                for file in os.listdir(os.path.join(EVALUATE_DIR, problem, student, TEST_SUBDIR)):
                    if not file.endswith(".c"):
                        continue
                    pool.apply_async(execure_only_worker, args=(problem, student, file, analyze_dict, buggy_pass_num_dict[problem][file] if TEST_FIXED else 0, lock))

        pool.close()
        pool.join()
        with open(LOG_FILE, "w") as f:
            json.dump(dict(analyze_dict), f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--pre-execute-needed', action="store_true")
    parser.add_argument('--label', action="store_true")
    parser.add_argument('--test-fixed', action="store_true")
    args = parser.parse_args()
    get_global_value(args.test, args.pre_execute_needed, args.label, args.test_fixed)
    if not LABEL:
        evalute_only()
    else:
        evalute_and_label() 