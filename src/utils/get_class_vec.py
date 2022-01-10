#!/usr/bin/env python
# encoding: utf-8
"""
@description: 从降维后的128维方法向量中得到类的128维向量
@date: 2021/12/17
"""
import json
import sys
import pandas as pd
import numpy as np


_input_file = sys.argv[1]
_class_file = sys.argv[2]
_output_path = sys.argv[3]
_class_set = set()
_class_dict = {}
_method_dict = {}
_result_dict = {}


def get_class_set():
    """
    将所有需要用的的类名存入set中
    """
    data = pd.read_csv(_class_file, header=0)  # 读取文件中所有数据
    for _class in data['class']:
        _class_set.add(_class)
    for method in data['method']:
        help_list = method.split('.')
        del help_list[-1]
        class_name = ''
        for i in range(0, len(help_list) - 1):
            class_name = class_name + help_list[i] + '.'
        class_name = class_name + help_list[-1]
        _class_set.add(class_name)
    print('The number of class: ' + str(len(_class_set)) + '\n')


def get_class_name(method_name):
    """
    获取一条方法记录的类名信息
    :param method_name: 方法向量名
    :return: 对应的类名
    """
    list = method_name.split('.')
    del list[-1]
    class_name = ''
    for el in list:
        class_name = class_name + el + '.'
    class_name = class_name.rstrip('.')
    return class_name


def get_class_dict():
    """
    构建同一类下的方法向量集
    """
    input_file = open(_input_file, 'r')
    _method_dict = json.load(input_file)
    for method_name, method_vec in _method_dict.items():
        class_name = get_class_name(method_name)
        if class_name in _class_set:
            if class_name in _class_dict:
                _class_dict.get(class_name).append(method_vec)
            else:
                vec_list = [method_vec]
                _class_dict[class_name] = vec_list


def get_class_vec():
    """
    计算每个类的类向量
    """
    output_file = open(_output_path + "\\" + _input_file.split("\\")[-1].split("_")[0] + "_class_vec.txt", "a+")
    for class_name, vec_list in _class_dict.items():
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': '{: 0.9f}'.format})  # 格式化输出，浮点数保留6位小数且不够位时右侧补0
        data = np.array(vec_list).astype(float)
        class_vec = np.mean(data, axis=0).tolist()
        _result_dict[class_name] = class_vec
    json.dump(_result_dict, output_file)
    # res = json.dumps(_result_dict)
    # output_file.write(res)
    # output_file.close()


if __name__ == "__main__":
    get_class_set()
    get_class_dict()
    get_class_vec()