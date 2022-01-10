import json
import sys
import os

read_path_0 = sys.argv[1]
read_path_1 = sys.argv[2]
output_path = sys.argv[3]
method = sys.argv[4]


def deal_files():
    files = os.listdir(read_path_0)
    for file in files:
        if os.path.splitext(file)[1] == '.json':
            
            file_name = os.path.splitext(file)[0]
            output_file = method + '_' + file_name + '.txt'
        
            with open(read_path_0 + "\\" + file, 'r', encoding='utf-8') as f0, open(read_path_1 + "\\" + output_file,
                    'r', encoding='utf-8') as f1, open(output_path + "\\" + output_file, 'w', encoding='utf-8') as f2:
                st = f0.read()
                s = json.loads(st)
                
                dict = {}
                count = 0
                for var in s["variables"]:
                    dict[count] = var
                    count += 1
                  
                next(f1)
                data = f1.read()
                data = data.rstrip()
                lines = [line.split(" ") for line in data.split("\n")]

                for line in lines:
                    if int(line[0]) <= count:
                        id = int(line[0])
                        line[0] = dict[id]
                        o = str(line)
                        o = o.replace("\\\\", "\\")
                        o = o.replace("\'", '')
                    f2.write(o + '\n')


if __name__ == "__main__":
    deal_files()
        
    
