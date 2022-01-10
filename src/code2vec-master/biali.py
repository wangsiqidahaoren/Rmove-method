import os


def show_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            filePath = os.path.splitext(file)
            if filePath[0] == '.java' or filePath[1] == '.java':
                all_files.append(file)
    print(all_files)
    return all_files


# 传入空的list接收文件名
contents = show_files("C:\\Users\\17471\\Desktop\\astminer-master\\src", [])
# 循环打印show_files函数返回的文件名列表
for content in contents:
    print(content)

print(len(contents))