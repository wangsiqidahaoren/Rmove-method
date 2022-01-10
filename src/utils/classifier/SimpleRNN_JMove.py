import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import datetime
import sys
import os
import pandas as pd

def loadCSVfile(filename):
    tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
    data = tmp[0:,0:-1].astype(np.float)#加载数据部分
    label = tmp[0:,-1:].astype(np.float)#加载类别标签部分
    # print(len(data))
    # print(len(label))
    return data, label #返回array类型的数据




def train_model(data,label):
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Embedding(input_dim=721,output_dim=64,input_length=256))
    model.add(tf.keras.layers.SimpleRNN(64, dropout=0.2, recurrent_dropout=0.5))
    # model.add(tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.5))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(np.expand_dims(data, 1), label, epochs=20, batch_size=32, validation_data=(np.expand_dims(data_train,1),label_train))
    # plt.plot(history.epoch, history.history.get('acc'), 'r', label='acc')
    # plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='val_acc')
    # plt.legend()
    # plt.show()
    return model






data, label = loadCSVfile("./train/CV_NV.csv")#训练集 训练集的路径

seed = 42
np.random.seed(seed)
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size=0.2,random_state=seed)
model = train_model(data_train, label_train)

with open("CV_NV_result.txt",'w') as file:
    print("---------------------test weka----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_weka.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test maven----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_maven.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test jtopen----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_jtopen.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test jmeter----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_jmeter.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test freemind----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_freemind.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test freecol----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_freecol.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test drjava----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_drjava.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
    # 新的测试集

    print("---------------------test ant----------------------",file=file)
    X_test, Y_test = loadCSVfile("./test/test_ant.csv")  # 测试集
    # predict probabilities for test set
    yhat_probs = model.predict(np.expand_dims(X_test, 1), verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(np.expand_dims(X_test, 1), verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(Y_test, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision,file=file)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall,file=file)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1,file=file)

    # print("--------------------------------------------------------------",file=file)
