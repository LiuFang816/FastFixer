decoder_set = ['utf-8', 'gb18030', 'ISO-8859-2', 'gb2312', 'gbk', 'error']


def read_file(file_path):
    for decoder in decoder_set:
        try:
            file = open(file_path, 'r', encoding=decoder)
            text = file.read().encode(encoding='utf-8', errors='replace').decode(encoding='utf-8')
            file.close()
            return text
        except ValueError:
            if decoder == 'error':
                raise Exception(f'{file_path} has no way to decode')
            continue
