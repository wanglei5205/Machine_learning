# -*- coding: utf-8 -*-
"""
###############################################################################
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 代码：http://github.com/wanglei5205
# 博客：http://cnblogs.com/wanglei5205
# 目的：学习使用GridSearchCV和StratifiedKFold对xgboost调参
# 官方API文档：http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
###############################################################################
"""
### load module
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from xgboost import plot_importance

### load datasets
digits = load_digits()

### data analysis
print(digits.data.shape)
print(digits.target.shape)

### data split 
x_train,x_test,y_train,y_test = train_test_split(digits.data,
                                                 digits.target,
                                                 test_size = 0.3,
                                                 random_state = 33)
### fit model for train data
model = XGBClassifier(learning_rate=0.1,
                      n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立树
                      colsample_btree=0.8,       # 随机算哦80%样本选择特征
                      objective='multi:softmax', # 指定损失函数
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27            # 随机数
                      )

model.fit(x_train,
          y_train,
          eval_set = [(x_test,y_test)],          # 评估数据集
          eval_metric = "mlogloss",              # 评估标准
          early_stopping_rounds = 10,            # 当loss有10次未变，提前结束评估
          verbose = False)                       # 显示提前结束


# 参数字典
param_grid = {'learning_rate':[0.05,0.1,0.25,0.3],
              'max_depth':range(2,10),
              'n_estimators':range(100,110,120)}

kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
grid_search = GridSearchCV(model,                   # 模型
                           param_grid,              # 待调参数（字典）
                           scoring="neg_log_loss",  # 模型评估准则
                           n_jobs=1,               # -1表示使用全部的cpu运算
                           cv=kfold)
grid_result = grid_search.fit(digits.data,digits.target)

print(grid_search.grid_scores_)
print(grid_search.best_score_)
print(grid_search.best_params_)

### summarize results
print("best:%f using %s" % (grid_result.best_score_,grid_result.best_params))
means = grid_result.cv_results_['mean_test_score']
params =  grid_result.cv_results_['params']

for mean, param in zip(means,params):
    print("%f with: %r" % (mean,param))

### plot feature importance
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

### make prediction for test data
y_pred = model.predict(x_test)

### model evaluate
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))
