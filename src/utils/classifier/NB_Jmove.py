#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score


# In[2]:


train_lists = ['CV_DW','CV_GR','CV_LN','CV_NV','CV_PN','CV_SN','CV_WL','CS_DW','CS_GR','CS_LN','CS_NV','CS_PN','CS_SN','CS_WL']
test_lists = ['weka','maven','jtopen','jmeter','freemind','freecol','drjava','ant']


# In[3]:


def load_data(train, test):
    tmp1 = train + '.csv'
    tmp2 = test + '.csv'
    df1 = pd.read_csv(tmp1,header=None)
    df2 = pd.read_csv(tmp2,header=None) 

    for u in df1.columns:
        if df1[u].dtype==bool:
            df1[u]=df1[u].astype('int')
        
    for u in df2.columns:
        if df2[u].dtype==bool:
            df2[u]=df2[u].astype('int')
    return df1,df2


# In[4]:


def train_test(df1,df2):
    Xtrain = df1.iloc[:,:-1]
    Xtest = df2.iloc[:,:-1]

    Ytrain = df1.iloc[:,-1]
    Ytest = df2.iloc[:,-1]
    
    return Xtrain,Xtest,Ytrain,Ytest


# In[5]:


if __name__ == '__main__':
    for train in train_lists:
        for test in test_lists:
            f = open('log.txt','a+')
            df1,df2 = load_data(train, test)
            Xtrain,Xtest,Ytrain,Ytest = train_test(df1,df2)
            clf = GaussianNB()
            clf = clf.fit(Xtrain, Ytrain)
            y_pred=clf.predict(Xtest)
            print("训练集：", train)
            print("测试集：", test)
            print('精确率：%.3f' % precision_score(Ytest, y_pred))
            print('召回率：%.3f' % recall_score(Ytest, y_pred))
            print('F1值：%.3f' % f1_score(Ytest, y_pred))
            f.write("训练集: " + train)
            f.write("         测试集：" + test + "\n")
            f.write('精确率：%.3f' % precision_score(Ytest, y_pred))
            f.write('       召回率：%.3f' % recall_score(Ytest, y_pred))
            f.write('       F1值：%.3f' % f1_score(Ytest, y_pred) + "\n")
            f.close()

