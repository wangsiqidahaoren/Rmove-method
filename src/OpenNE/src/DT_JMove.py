from sklearn.model_selection import train_test_split  # 划分测试集与训练集
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 导入测试数据数据集
print(index)
dt = np.dtype((np.float64, (1, 128)))
my_data = np.loadtxt('datas/CV_NV.txt', comments=None, delimiter='#', usecols=np.arange(0, 256), dtype=float)
my_label = np.loadtxt('datas/CV_NV.txt', comments=None, delimiter='#', usecols=(256,), dtype=int)
# 划分测试集-训练集-验证集
train_x, test_x, train_y, test_y = train_test_split(my_data, my_label, test_size=0.2, shuffle=True,
                                                    random_state=42)
# CV
# DeepWalk 6 1 2
# GraRep 5 9 2
# Line 5 1 2
# node2vec 8 1 2
# ProNE 4 5 2
# SDNE 7 3 2
# Walklets 9 2 2
# CS
# DeepWalk 9 1 2
# GraRep 9 4 2
# Line 9 1 2
# node2vec 6 5 2
# ProNE 7 6 2
# SDNE 6 1 2
# Walklets 6 1 2

f1 = 0
bm = 0
bl = 0
bs = 0
for max_depth in range(1, 10, 1):
    for min_samples_leaf in range(1, 10, 1):
        for min_samples_split in range(2, 10, 1):
            my_model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                   random_state=42)
            # 模型训练
            my_model.fit(train_x, train_y)
            # 模型预测
            y_pred = my_model.predict(test_x, check_input=True)
            if f1_score(test_y, y_pred) > f1:
                f1 = f1_score(test_y, y_pred)
                bm = max_depth
                bl = min_samples_leaf
                bs = min_samples_split
print(bm, bl, bs)
my_model = tree.DecisionTreeClassifier(max_depth=bm, min_samples_leaf=bl, min_samples_split=bs, random_state=42)
# 模型训练
my_model.fit(train_x, train_y)
# 模型预测
fileNames = ["ant", "drjava", "freecol", "freemind", "jmeter", "jtopen", "maven", "weka"]
for fileName in fileNames:
    print(fileName, end=': ')
    test_x = np.loadtxt('T11/datas/' + fileName + '.txt', comments=None, delimiter='#', usecols=np.arange(0, 256),
                        dtype=float)
    test_y = np.loadtxt('T11/datas/' + fileName + '.txt', comments=None, delimiter='#', usecols=(256,), dtype=int)
    y_pred = my_model.predict(test_x, check_input=True)
    print("P:", end=' ')
    print(precision_score(test_y, y_pred), end=' ')
    print("R:", end=' ')
    print(recall_score(test_y, y_pred), end=' ')
    print("F1:", end=' ')
    print(f1_score(test_y, y_pred))
