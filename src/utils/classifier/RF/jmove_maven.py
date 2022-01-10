import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


data = pd.read_csv('../data/cs/CS_DW.csv')

data.flag = data.flag.astype(str).map({"False": 0, "True": 1})
y = data.flag
# print(y)
x = data.drop('flag',axis=1)
# print(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=400,min_samples_leaf=10,min_samples_split=20,max_depth=11,random_state=42,
                             criterion='gini',class_weight=None)

rfc.fit(xtrain, ytrain)

maven_data = pd.read_csv("../data/jmove/test_maven.csv")
maven_data.flag = maven_data.flag.astype(str).map({"False": 0, "True": 1})

maven_y = maven_data.flag
maven_x = maven_data.drop('flag',axis=1)


p = precision_score(list(maven_y),rfc.predict(maven_x))
r = recall_score(list(maven_y),rfc.predict(maven_x))
f1 = f1_score(list(maven_y),rfc.predict(maven_x))

print(p)
print(r)
print(f1)
