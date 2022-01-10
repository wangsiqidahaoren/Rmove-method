#!/usr/bin/env python
# encoding: utf-8
"""
@description: 用GRU训练并对JMove数据进行测试
@date: 2021/12/20
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers import scikit_learn
import sys
import datetime


_training_data_path = sys.argv[1]  # 训练数据集目录
_JMove_project_path = sys.argv[2]  # 测试数据集目录


def create_model():
    # 创建模型
    model = Sequential()
    model.add(GRU(4, input_shape=(1, 256)))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def load_data(src):
    """
    加载训练数据集并划分训练集和测试集
    :param src: 训练集文件
    :return: 训练集和测试集的vec和label
    """
    source_data = pd.read_csv(src, header=None)
    data_list = source_data.values.tolist()
    vec_list = []
    labels = []
    for el in data_list:
        if el[-1]:
            labels.append(1)
        else:
            labels.append(0)
        el.pop()
        vec = el
        vec_list.append(vec)
    vec_list = np.asarray(vec_list)
    labels = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(vec_list, labels, test_size=0.20, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train, X_test, y_train, y_test


def test(estimator, path, trainingfile, info):
    """
    分别测试测试目录下的所有项目向量文件并写入对应的结果文件中
    :param estimator: 训练得到的最优模型
    :param path: 测试集目录
    :param trainingfile: 训练模型所使用的训练数据集，这里仅用于对输出结果文件命名
    :return: None
    """
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = open(path + "\\" + file)  # 打开文件
            JMove_data = pd.read_csv(f, header=None).values.tolist()
            JMove_vec = []
            JMove_label = []
            for el in JMove_data:
                if el[-1]:
                    JMove_label.append(1)
                else:
                    JMove_label.append(0)
                el.pop()
                vec = el
                JMove_vec.append(vec)
            f.close()
            JMove_vec = np.asarray(JMove_vec)
            JMove_vec = np.reshape(JMove_vec, (JMove_vec.shape[0], 1, JMove_vec.shape[1]))
            testing_project = file.split(".")[0]
            output = open("JMove_Test_" + trainingfile.split(".")[0] + ".txt", "a+")
            # 测试数据
            print("Start testing : " + testing_project + "\n")
            predict_res = estimator.predict(JMove_vec)
            precision = '精确率：%.3f' % precision_score(JMove_label, predict_res)
            recall = '召回率：%.3f' % recall_score(JMove_label, predict_res)
            f1 = 'F1值：%.3f' % f1_score(JMove_label, predict_res)
            output.write(
                "-----------------------------------------------------------------------------------------------\n")
            output.write("Testing data : " + testing_project + "\n")
            output.write("Testing time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
            output.write(info + "\n")
            output.write(precision + "\n")
            output.write(recall + "\n")
            output.write(f1 + "\n")
            output.write(
                "-----------------------------------------------------------------------------------------------\n\n\n")
            output.close()


def train(X_train, X_test, y_train, y_test, trainingfile):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param trainingfile:
    :return:
    """
    model = scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)
    # 参数列表，可自己微调
    tuned_parameters = [{'batch_size': [1, 8, 16, 64], 'epochs': [10, 30, 50, 70, 100]}]
    # 生成模型
    grid = GridSearchCV(model, param_grid=tuned_parameters, cv=5, scoring="roc_auc", verbose=2)
    grid_result = grid.fit(X_train, y_train)
    # 把数据交给模型训练
    cls = grid_result.best_estimator_
    cls.fit(X_train, y_train)
    info = "The best parameters are %s with a score of %0.3f" % (grid_result.best_params_, grid_result.best_score_)
    test(cls, _JMove_project_path, trainingfile, info)


if __name__ == '__main__':
    files = os.listdir(_training_data_path)  # 得到训练文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            training_file = open(_training_data_path + "\\" + file)  # 打开文件
            X_train, X_test, y_train, y_test = load_data(training_file)
            train(X_train, X_test, y_train, y_test, file)
