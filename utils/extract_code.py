def extract_code_in_markdown_format(markdown_text):
    """
    Extract code in markdown format.
    """
    code = ''
    in_code_block = False
    for line in markdown_text.split('\n'):
        if line.startswith('```'):
            if in_code_block:
                in_code_block = False
            else:
                in_code_block = True
        elif in_code_block:
            code += line + '\n'
    return code if code != '' else None


def extract_code_embrace_by_label(text, label):
    code = ''
    in_code_block = False
    for line in text.split('\n'):
        if line.startswith('[{}]'.format(label)):
            in_code_block = True
        elif line.startswith('[/{}]'.format(label)):
            in_code_block = False
        elif in_code_block:
            code += line + '\n'
    return code if code != '' else None