#!/usr/bin/env python
# encoding: utf-8
"""
@description: 用XGB训练并对JMove数据进行测试
@date: 2021/12/18
"""
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sys
import os
import datetime
import numpy as np

_training_data_path = sys.argv[1]  # 训练数据集目录
_JMove_project_path = sys.argv[2]  # 测试数据集目录


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
    vec_list = np.array(vec_list)
    X_train, X_test, y_train, y_test = train_test_split(vec_list, labels, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def test(estimator, path, trainingfile):
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
            JMove_vec = np.array(JMove_vec)
            testing_project = file.split(".")[0]
            output = open("JMove_Test_" + trainingfile.split(".")[0] + ".txt", "a+")
            # 测试数据
            print("Start testing : " + testing_project + "\n")
            y_pred = estimator.predict(JMove_vec)
            predictions = [round(value) for value in y_pred]
            precision = precision_score(JMove_label, predictions)
            recall = recall_score(JMove_label, predictions)
            f1 = f1_score(JMove_label, predictions)
            accuracy = accuracy_score(JMove_label, predictions)

            # precision = '精确率：%.3f' % precision_score(JMove_label, predict_res)
            # recall = '召回率：%.3f' % recall_score(JMove_label, predict_res)
            # f1 = 'F1值：%.3f' % f1_score(JMove_label, predict_res)
            output.write(
                "-----------------------------------------------------------------------------------------------\n")
            output.write("Testing data : " + testing_project + "\n")
            output.write("Testing time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
            output.write("accuracy: %.2f%%" % (accuracy * 100.0) + "\n")
            output.write("precision: {:.3}, recall: {:.3}, f1:{:.3}".format(precision, recall, f1) + "\n")
            # output.write(accuracy + "\n")
            # output.write(precision + "\n")
            # output.write(recall + "\n")
            # output.write(f1 + "\n")
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
    # 网格搜索参数列表
    # tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [1/128, 1/256, 1e-2, 1e-3, 1e-4],
    #                      'C': [0.1, 0.4, 0.7, 1, 10, 100, 500, 1000]},
    #                     {'kernel': ['linear'], 'C': [0.1, 0.4, 0.7, 1, 10, 100, 500, 1000]}]
    model = XGBClassifier(learning_rate = 0.1, n_estimators = 100, max_depth = 9, min_child_weight=1,
                          subsample = 0.6, colsample_bytree = 0.8, gamma = 0.4, reg_alpha = 1, reg_lambda = 0.01)
    # param_grid = {
    #     'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    #     'n_estimators': [i for i in range(100, 1000, 100)],
    #     'max_depth': [i for i in range(3, 10, 1)],
    #     'min_child_weight': [i for i in range(1, 6, 1)],
    #     'gamma': [i / 10.0 for i in range(0, 5)],
    #     'subsample': [i / 10.0 for i in range(6, 10)],
    #     'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    #     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    #     'reg_lambda': [1e-5, 1e-2, 0.1, 1, 2, 100]
    # }
    # 生成模型
    print("Start trainging : " + trainingfile + "\n")
    # grid = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=5, scoring="roc_auc", verbose=2)
    # grid = GridSearchCV(model, param_grid, n_jobs = 4, iid = False, cv = 5)
    # grid.fit(X_train, y_train)
    # cls = grid.best_estimator_
    # 把数据交给模型训练
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # cls.fit(X_train, y_train)
    test(model, _JMove_project_path, trainingfile)


if __name__ == '__main__':
    files = os.listdir(_training_data_path)  # 得到训练文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            training_file = open(_training_data_path + "\\" + file)  # 打开文件
            X_train, X_test, y_train, y_test = load_data(training_file)
            train(X_train, X_test, y_train, y_test, file)
