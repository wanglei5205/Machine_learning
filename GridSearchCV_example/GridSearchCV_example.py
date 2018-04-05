# -*- coding: utf-8 -*-
"""
# 数据：20类新闻文本
# 模型：svc
# 调参：gridsearch
"""
### 加载模块
import numpy as np
import pandas as pd

### 载入数据
from sklearn.datasets import fetch_20newsgroups                          # 20类新闻数据
news = fetch_20newsgroups(subset='all')                                  # 生成20类新闻数据

### 数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data[:300],
                                                    news.target[:300],
                                                    test_size=0.25,      # 测试集占比25%
                                                    random_state=33)     # 随机数 
### pipe-line
from sklearn.feature_extraction.text import TfidfVectorizer              # 特征提取
from sklearn.svm import SVC                                              # 载入模型
from sklearn.pipeline import Pipeline                                    # pipe_line模式
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')),
                ('svc', SVC())])

### 网格搜索
from sklearn.model_selection import GridSearchCV
parameters = {'svc__gamma': np.logspace(-1, 1)}                           # 参数范围（字典类型）

gs = GridSearchCV(clf,          # 模型
                  parameters,   # 参数字典
                  n_jobs=1,     # 使用1个cpu
                  verbose=0,    # 不打印中间过程
                  cv=5)         # 5折交叉验证

gs.fit(X_train, y_train)        # 在训练集上进行网格搜索

### 最佳参数在测试集上模型分数
print("best:%f using %s" % (gs.best_score_,gs.best_params_))

### 测试集下的分数
print("test datasets score" % gs.score(X_test, y_test))

### 模型不同参数下的分数
# 方式一(0.20版本将删除)
print(gs.grid_scores_)

# 方式二(0.20推荐的方式)
means = gs.cv_results_['mean_test_score']
params =  gs.cv_results_['params']

for mean, param in zip(means,params):
    print("%f with: %r" % (mean,param))