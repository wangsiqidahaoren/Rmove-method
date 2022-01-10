
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
# 处理数据的库
import numpy as np
import sklearn
import pandas as pd
from pathlib import Path
import csv
from sklearn.model_selection import train_test_split
# 系统库
import os
import sys
import time
# TensorFlow的库
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,LSTM,Bidirectional, Dropout
import keras
import keras_metrics as km
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import GridSearchCV

"""
说明：训练集为train_test_spilt分出来的，测试集为data/train/JMove_test_data中的
"""

#处理数据
trainset = "data/train/CS/CS_WL.csv"                #训练集路径
testset_dir = "data/train/JMove_test_data/"          #测试集路径
X = []          #向量
y = []          #标签

with open(trainset) as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        a = row[0:256]
        b = []
        for i in a:
            b.append(float(i))
        X.append(b)
        if row[256] == "True":
            y.append(1)
        elif row[256] == "False":
            y.append(0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
print(X_train.shape)

#构建单层LSTM模型
model = keras.models.Sequential()

'''
model.add(LSTM(units=64,activation='tanh',return_sequences = False,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dense(units=64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='binary_crossentropy',
              metrics=[km.f1_score(), km.binary_precision(), km.binary_recall()])
'''


model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=['acc',km.f1_score(), km.binary_precision(), km.binary_recall()])



#训练模型
n_epochs = 20           #训练轮数
"""
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),
                    validation_freq=1)
"""
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=n_epochs)

model.summary()

#计算准确率等
p = Path(testset_dir)
filelist = list(p.glob("*.csv"))

for file in filelist:
    X_JMove = []
    y_JMove = []
    #print(file)
    filename = os.path.splitext(file)[0]
    filename = os.path.basename(filename)
    print(filename)
    with open(file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            a = row[0:256]
            b = []
            for i in a:
                b.append(float(i))
            X_JMove.append(b)
            if row[256] == "True":
                y_JMove.append(1)
            elif row[256] == "False":
                y_JMove.append(0)
        X_JMove = np.array(X_JMove)
        y_JMove = np.array(y_JMove)
        X_JMove = X_JMove.reshape(X_JMove.shape[0], 1, X_JMove.shape[1])
        y_pred = model.predict(X_JMove, verbose=0)
        y_pred = np.int64(y_pred > 0.5)
        accuracy = accuracy_score(y_JMove, y_pred)
        precision = precision_score(y_JMove, y_pred)
        recall = recall_score(y_JMove, y_pred)
        f1 = f1_score(y_JMove, y_pred)
        #print(filename + "accuracy:" + accuracy + ",precision:" + precision + ",recall:" + recall + ",F1_score:" + f1_score)
        print(filename + ":")
        print("accuracy:\t%.3f"%accuracy)
        print("precision:\t%.3f"%precision)
        print("recall:\t%.3f"%recall)
        print("f1_score:\t%.3f"%f1)
        print("---------------------------------------------------------------")

"""
y_pred = model.predict(X_test, verbose=0)
#print("Hello_______________________")
#print(y_pred)
#print("Hi--------------------")
y_pred = np.int64(y_pred>0.5)
#print(y_pred)
#print("Hey-----------------")
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
"""
"""
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)
"""

"""
#画图
plt.plot(history.history['loss']    , label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
"""







