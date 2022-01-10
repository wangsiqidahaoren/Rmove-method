import json
import os

read_path = ""
output_path = ""


def get_file_path():
    read_path = r"path"
    output_path = r"path"
    return read_path, output_path


def deal_files():
    files = os.listdir(read_path)
    for file in files:
        if os.path.splitext(file)[1] == '.json':
            file_name = os.path.splitext(file)[0]
            f = open(read_path + "\\" + file, 'r', encoding='utf-8')
            output_file = file_name + '.txt'
            t = open(output_path + "\\" + output_file, 'w')
            str = f.read()
            s = json.loads(str)
            for cell in s["cells"]:
                print (cell["src"], cell["dest"], sep='  ', file=t)
            t.close()
            

if __name__ == "__main__":
    read_path, output_path = get_file_path()
    deal_files()
