import json
import os
read_path = ""
output_path = ""


def get_file_path():
    read_path = r"G:\weixin\documents\WeChat Files\wxid_5ak6f92qymxe22\FileStorage\File\2021-11"
    output_path = r"G:\depends-0.9.6\txt1"
    return read_path, output_path


def deal_files():
    files = os.listdir(read_path)
    for file in files:
        if os.path.splitext(file)[1] == '.json':
            file_name = os.path.splitext(file)[0]
            f = open(read_path + "\\" + file, 'r', encoding='utf-8')
            output_file = file_name + '.txt'
            t = open(output_path + "\\" + output_file, 'w', encoding='utf-8')
            str = f.read()
            s = json.loads(str)
            count = 0
            for var in s["variables"]:
                print(count, var, file=t)
                count += 1
            t.close()
            

if __name__ == "__main__":
    read_path, output_path = get_file_path()
    deal_files()
