# -*- coding: utf-8 -*-
"""
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 博客：http://cnblogs.com/wanglei5205
# github：http://github.com/wanglei5205
"""
### 载入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### 载入数据
from sklearn.datasets import load_digits
digits = load_digits()

### 数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(digits.data,
                                                 digits.target,
                                                 test_size = 0.3,
                                                 random_state = 33)

### 载入模型
from xgboost import XGBClassifier
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

### 网格搜素
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':[100,110,120,130],
              'max_depth':[2,3,4,5,6,]}
gs = GridSearchCV(xgb1,
                  param_grid,
                  scoring='accuracy',
                  cv=3,
                  verbose=0,
                  n_jobs=1)

gs.fit(x_train,y_train)
print(gs.best_params_,gs.best_score_)

"""
xgb1.fit(x_train,y_train)

### 模型预测
y_pred = xgb1.predict(x_test)

### 性能评估
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
print(acc)
"""
