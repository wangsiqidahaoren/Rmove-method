#!/usr/bin/env python
# encoding: utf-8
"""
@description: 将两种方法得到的方法向量进行拼接并利用PCA降维至128维
@date: 2021/12/17
"""

import re
from sklearn.decomposition import PCA
import numpy as np
import json
import sys
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=10000, formatter={'float': '{:0.9f}'.format})
pca = PCA(n_components=128)


_code2vec_file = sys.argv[1]
_openne_file = sys.argv[2]
_output_path = sys.argv[3]
_project_name = sys.argv[4]
_method_name = []
_method_vec = []
_code2vec_dict = {}
_method_dict = {}
_result_dict = {}
# _openne_dict = {}


# def get_openne_dict():
#     file = open(_openne_file, 'r', encoding='utf-8')
#     lines = file.readlines()
#     for line in lines:
#         p1 = re.compile(r'[(](.*?)[)]', re.S)  # 最小匹配
#         method_name = re.findall(p1, line)[0]  # 获取每个方向向量的信息（类名+方法名）
#         vec = line.strip().rstrip(']').replace(' ', '').split(',')
#         del vec[0]
#         _openne_dict[method_name] = vec
#     file.close()


def get_code2vec_dict():
    file = open(_code2vec_file, 'r', encoding='utf-8')
    for line in file.readlines():
        line = line.strip()
        ls2 = line.split('Desktop\\')
        class_name = ls2[1].split('_')[0]
        method_name = ls2[1].split('_')[1].split(' ')[0][0].lower() + ls2[1].split('_')[1].split(' ')[0][1:]
        vec = ls2[1].replace(ls2[1].split(' ')[0], '').split(' ')
        del vec[0]
        info = class_name + ',' + method_name
        if info not in _code2vec_dict:
            _code2vec_dict[info] = vec
    file.close()


def concat_vec():
    file = open(_openne_file, 'r', encoding='utf-8')
    lines = file.readlines()
    for line in lines:
        word = 'Desktop\\'
        ls2 = line.split(word)
        class_name = ls2[1].split('(')[0]
        method_name = line.split(')')[0].rsplit('.', 1)[1]
        info = class_name + ',' + method_name
        if info in _code2vec_dict:
            p1 = re.compile(r'[(](.*?)[)]', re.S)  # 最小匹配
            result_info = re.findall(p1, line)[0]  # 获取方法向量的信息（类名+方法名）
            vec = line.strip().rstrip(']').replace(' ', '').split(',')
            del vec[0]
            vec = _code2vec_dict.get(info) + vec
            _method_dict[result_info] = vec
    file.close()


def load_data():
    for name, vec in _method_dict.items():
        _method_name.append(name)
        _method_vec.append(vec)
    print("The number of method_vec: " + str(len(_method_vec)))


def pca_vec():
    pca_data = np.array(_method_vec)
    pca_data = pca_data.astype(np.float32)
    pca_data = pca.fit_transform(pca_data)
    output_file = _output_path + '\\' + _project_name + '_method_vec.txt'
    result_file = open(output_file, 'w')
    for i in range(0, len(pca_data)):
        _result_dict[_method_name[i]] = pca_data[i].tolist()
    json.dump(_result_dict, result_file)
    result_file.close()


if __name__ == '__main__':
    get_code2vec_dict()
    concat_vec()
    load_data()
    pca_vec()
