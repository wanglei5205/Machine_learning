# -*- coding: utf-8 -*-
"""
###############################################################################
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 代码：http://github.com/wanglei5205
# 博客：http://cnblogs.com/wanglei5205
# 目的：学习xgboost的XGBClassifier函数
# 官方API文档：http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
###############################################################################
"""
### load module
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

### load datasets
digits = datasets.load_digits()

### data analysis
print(digits.data.shape)
print(digits.target.shape)

### data split 
x_train,x_test,y_train,y_test = train_test_split(digits.data,
                                                 digits.target,
                                                 test_size = 0.3,
                                                 random_state = 33)

### fit model for train data
# fit函数参数：eval_set=[(x_test,y_test)]  评估数据集,list类型
# fit函数参数：eval_metric="mlogloss"      评估标准(多分类问题，使用mlogloss作为损失函数)
# fit函数参数：early_stopping_rounds= 10   如果模型的loss十次内没有减小，则提前结束模型训练
# fit函数参数：verbose = True              True显示，False不显示
model = XGBClassifier()
model.fit(x_train,
          y_train,
          eval_set = [(x_test,y_test)],  # 评估数据集    
          
          eval_metric = "mlogloss",
          early_stopping_rounds = 10,
          verbose = True)

### make prediction for test data
y_pred = model.predict(x_test)                   

### model evaluate 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))
"""
95.0%
"""