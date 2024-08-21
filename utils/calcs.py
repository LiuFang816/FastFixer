import re
import numpy as np

def edit_distance(wrong_code, correct_code):
    """
    Calculate the edit distance between two codes.
    """
    wrong_code = wrong_code.split('\n')
    correct_code = correct_code.split('\n')
    wrong_code = [re.sub('\s', '', line) for line in wrong_code if line != '']
    correct_code = [re.sub('\s', '', line) for line in correct_code if line != '']
    len_wrong_code = len(wrong_code)
    len_correct_code = len(correct_code)
    dp = [[0] * (len_correct_code + 1) for _ in range(len_wrong_code + 1)]
    for i in range(len_wrong_code + 1):
        dp[i][0] = i
    for j in range(len_correct_code + 1):
        dp[0][j] = j
    for i in range(1, len_wrong_code + 1):
        for j in range(1, len_correct_code + 1):
            if wrong_code[i - 1] == correct_code[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1
                )
    return dp[len_wrong_code][len_correct_code]


def semantic_similarity(correct_code):
   return analyze_relationships(correct_code)[:2]


import re

def extract_variables(line):
    # 简单的变量提取方法，适用于基本情况
    # 可以根据需要扩展
    variables = re.findall(r'\b[A-Za-z_]\w*\b', line)
    return set(variables)

def extract_function_calls(line):
    # 提取函数调用
    function_calls = re.findall(r'\b([A-Za-z_]\w*)\s*\(', line)
    return set(function_calls)

def is_assignment(line):
    # 判断是否为赋值操作
    return '=' in line and not line.strip().startswith('//')

def analyze_code(code):
    lines = code.split('\n')
    variables_by_line = []
    function_calls_by_line = []
    
    for line in lines:
        variables_by_line.append(extract_variables(line))
        function_calls_by_line.append(extract_function_calls(line))
    
    return variables_by_line, function_calls_by_line

def analyze_relationships(code):
    variables_by_line, function_calls_by_line = analyze_code(code)
    lines = code.split('\n')
    
    relationships = []
    
    for i, line_a in enumerate(lines):
        for j, line_b in enumerate(lines):
            if i == j:
                continue
            
            # 判断a行是否对b行使用的变量进行了赋值操作
            if is_assignment(line_a):
                assigned_variables = extract_variables(line_a.split('=')[0])
                if assigned_variables & variables_by_line[j]:
                    relationships.append((i+1, j+1, 'assignment'))
            
            # 判断a行是否是b行调用的函数中的一行代码
            called_functions = function_calls_by_line[j]
            for func in called_functions:
                if re.search(r'\b' + func + r'\b', line_a):
                    relationships.append((i+1, j+1, 'function_call'))
    
    return np.asarray(relationships[:, :2]).tolist()


