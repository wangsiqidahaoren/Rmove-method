#!/usr/bin/env python
# encoding: utf-8
"""
@description: 将同一算法不同项目的training_data整合为一个大的训练集
@date: 2021/12/17
"""
import json
import csv
import os
import sys


path = sys.argv[1]  # training_data文件夹目录
output_path = sys.argv[2]
output_file = sys.argv[3]
f_out = open(output_path + '\\' + output_file + '.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f_out)
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        f = open(path + "\\" + file)  # 打开文件
        data = json.load(f)
        data_value = list(data.values())
        for val in data_value:
            csv_writer.writerow(val)
        f.close()
f_out.close()
