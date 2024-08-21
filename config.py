import os
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

# 项目根目录
BASE_DIR = "/home/FastFixer/FastFixer"
# 数据文件夹
DATA_DIR = os.path.join(BASE_DIR, "data")
# 记录文件夹
RES_DIR = os.path.join(BASE_DIR, "res")
# 模型修复后代码执行记录文件夹
MODEL_FIX_RES_DIR = os.path.join(RES_DIR, "fix")
# 错误代码执行记录记录文件夹
BUGGY_RES_DIR = os.path.join(RES_DIR, "buggy")
# 统计结果记录文件夹
STATISTIC_RES_DIR = os.path.join(RES_DIR, "statistic")
# 学生作业运行的工作目录
WORKSPACE = os.path.join(BASE_DIR, "tmp")
# 提取训练数据临时存放文件夹
EXTRACT_DIR = os.path.join(DATA_DIR, "tmp")
# 过滤后的训练数据存放文件夹
FILTERED_DIR = os.path.join(DATA_DIR, "filtered")

"""
change dataset here:
itsp_dataset
dataset(data structure homework)
filtered(all data structure homework)
"""

"""
itsp&dataset is form as follows:
data
├── itsp_dataset
│   ├── problemID_1
│   │   ├── buggy
│   │   │   ├── userID_1_buggy.c
│   │   │   ├── userID_2_buggy.c
│   │   │   └── ...
│   │   ├── correct
│   │   │   ├── userID_1_correct.c
│   │   │   ├── userID_2_correct.c
│   │   │   └── ...
│   │   ├── fixed
│   │   │   ├── userID_1_fixed.c
│   │   │   ├── userID_2_fixed.c
│   │   │   └── ...
│   │   └── testcase
│   ├── problemID_2
│   ├── ...

filtered is form as follows:
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

TRAIN_DATASET is the dataset used to train the model
FIX_DATASET is the dataset used to test the model
"""
TRAIN_DATASET = "filtered"
FIX_DATASET = "dataset"

# 训练数据源路径
TRAIN_DATA_DIR = os.path.join(DATA_DIR, TRAIN_DATASET)
# 修复数据源路径
FIX_DATA_DIR = os.path.join(DATA_DIR, FIX_DATASET)

# 存放用于训练的数据集的文件夹
FINETUNE_DATA_DIR = os.path.join(DATA_DIR, "finetune_data")
TOKENIZED_DATA_DIR = os.path.join(DATA_DIR, "tokenized_data")
# 测试用例文件夹名
TESTCASE_DIR = "testcase"

class FINETUNE_SHARED:
    MODEL = "codellama-instrcut-7b-hf-huggingface-with-buggy-reason"
    IS_FINETUNED = True
    FINETUNE_METHOD = "lora_8k_length_4000_knowledge_mask_mark_random_between_0_and_0_8_take_2"
    FIXED_DIR = (MODEL + FINETUNE_METHOD) if IS_FINETUNED else MODEL
    FINETUNE_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", FIXED_DIR)

CORRECT = 0
COMPILE_ERROR = 1
RUNTIME_ERROR = 2
WRONG_ANSWER = 3
TIME_LIMIT_EXCEEDED = 4
TIMEOUT = 1

BUGGY_CASE_PASS_NUM_FILE = os.path.join(STATISTIC_RES_DIR, "test_buggy_pass_num.json")
BUGGY_LABEL_FILE = os.path.join(STATISTIC_RES_DIR, "buggy_analyze_test.json")
TEST_BUGGY_LABEL_FILE = os.path.join(STATISTIC_RES_DIR, "buggy_analyze_test.json")
TRAIN_BUGGY_LABEL_FILE = os.path.join(STATISTIC_RES_DIR, "buggy_analyze_train.json")
# BUGGY_LABEL_FILE = os.path.join(STATISTIC_RES_DIR, "buggy_analyze_train.json")

PROXIES = {
    # "http": "http://192.168.205.123:7890",
    # "https": "http://192.168.205.123:7890",
}

"""题目相关，记录题目的属性"""

problem_properties = {
    "4778": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4656": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4779": {
        "is_output_file": False,
        "is_input_file": True,
        "has_extra_input": False,
        "input_file_name": "article.txt",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4642": {
        "is_output_file": False,
        "is_input_file": True,
        "has_extra_input": False,
        "input_file_name": "article.txt",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4644": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4643": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4559": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": ""
    },
    "4795": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4670": {
        "is_output_file": False,
        "is_input_file": True,
        "has_extra_input": False,
        "input_file_name": "example.c",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4672": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "5307": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "input.txt",
    },
    "6757": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4565": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": ""
    },
    "6744": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4564": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4563": {
        "is_output_file": True,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "fileout.txt",
        "extra_input_file_name": "filein.txt",
    },
    "4593": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "6882": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "bgstations.txt",
    },
    "4937": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": ""
    },
    "4935": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4811": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4808": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "dictionary3000.txt",
    },
    "4958": {
        "is_output_file": True,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "out.txt",
        "extra_input_file_name": "in.txt",
    },
    "4833": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4832": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4946": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4617": {
        "is_output_file": True,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "ordered.txt",
        "extra_input_file_name": "books.txt",
    },
    "4605": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": ""
    },
    "4602": {
        "is_output_file": True,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "output.txt",
        "extra_input_file_name": "encrypt.txt",
    },
    "4757": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4512": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "4620": {
        "is_output_file": True,
        "is_input_file": False,
        "has_extra_input": True,
        "input_file_name": "",
        "output_file_name": "in_crpyt.txt",
        "extra_input_file_name": "in.txt",
    },
    "10827": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
    "11443": {
        "is_output_file": False,
        "is_input_file": False,
        "has_extra_input": False,
        "input_file_name": "",
        "output_file_name": "",
        "extra_input_file_name": "",
    },
}
