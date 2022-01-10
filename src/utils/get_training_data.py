#!/usr/bin/env python
# encoding: utf-8
"""
@description: 根据映射文件拼接方法向量和对应的类向量并打上标签
@date: 2021/12/17
"""

import sys
import csv
import json


_mapping_file = sys.argv[1]
_class_file = sys.argv[2]
_method_file = sys.argv[3]
_output_path = sys.argv[4]
_project_name = _class_file.split("\\")[-1].split("_")[0]
_class_dict = {}
_method_dict = {}
_true_label = {}
_false_label = {}
_result = {}


def get_mapping(src):
    csvfile = open(src, "r")
    reader = csv.DictReader(csvfile)
    for row in reader:
        _true_label[row['method']] = row['class']
        help_list = row['method'].split('.')
        del help_list[-1]
        class_name = ''
        for i in range(0, len(help_list) - 1):
            class_name = class_name + help_list[i] + '.'
        class_name = class_name + help_list[-1]
        _false_label[row['method']] = class_name
    csvfile.close()


def process():
    class_file = open(_class_file, "r")
    # class_str = class_file.read()
    # _class_dict = json.loads(class_str)
    _class_dict = json.load(class_file)
    class_file.close()
    method_file = open(_method_file, "r")
    _method_dict = json.load(method_file)
    method_file.close()
    for moved_method, moved_class in _true_label.items():
        if moved_method in _method_dict.keys() and moved_class in _class_dict.keys():
            print(moved_method)
            key = moved_method + "->" + moved_class
            vec = _method_dict.get(moved_method) + _class_dict.get(moved_class)
            vec.append(True)
            _result[key] = vec
    print("=====================================================================================")
    for moved_method, moved_class in _false_label.items():
        if moved_method in _method_dict.keys() and moved_class in _class_dict.keys():
            print(moved_method)
            key = moved_method + "->" + moved_class
            vec = _method_dict.get(moved_method) + _class_dict.get(moved_class)
            vec.append(False)
            _result[key] = vec
    res_file = open(_output_path + "\\" + _project_name + "_training_data.txt", "w")
    print("The number of training_data: " + str(len(_result)) + '\n')
    json.dump(_result, res_file)
    res_file.close()


if __name__ == "__main__":
    get_mapping(_mapping_file)
    process()
