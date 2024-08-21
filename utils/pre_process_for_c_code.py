def preprocess_source_fsm(source_file, output_file):
    with open(source_file, 'r') as file:
        content = file.read()

    state = 'normal'
    i = 0
    length = len(content)
    output = []

    while i < length:
        if state == 'normal':
            if i+1 < length and content[i:i+2] == '//':
                state = 'single_line_comment'
                i += 1  # Skip the slash to stay at the next character
            elif i+1 < length and content[i:i+2] == '/*':
                state = 'multi_line_comment'
                i += 1  # Skip the star to stay at the next character
            elif content[i] == '#':
                # Check if this is a directive and ensure not inside string
                if content[i:].startswith('#include') or content[i:].startswith('#define'):
                    state = 'preprocessor'
                else:
                    output.append(content[i])
            else:
                output.append(content[i])
            if content[i] == '\n' and state == 'normal':
                # output.append('\n')
                pass

        elif state == 'single_line_comment':
            if content[i] == '\n':
                state = 'normal'
                output.append('\n')  # Keep the newline

        elif state == 'multi_line_comment':
            if i+1 < length and content[i:i+2] == '*/':
                state = 'normal'
                i += 1  # Skip the closing slash
            if content[i] == '\n':
                output.append('\n')  # Keep the newline to preserve line numbers

        elif state == 'preprocessor':
            if content[i] == '\n':
                state = 'normal'
                output.append('\n')  # Replace the directive with a newline

        i += 1

    with open(output_file, 'w') as fout:
        fout.write(''.join(output))

# 使用这个函数处理你的文件
preprocess_source_fsm('tmp.c', 'ttmp.c')
