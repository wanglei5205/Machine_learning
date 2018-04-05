# -*- coding: utf-8 -*-
"""
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 博客：http://cnblogs.com/wanglei5205
# github：http://github.com/wanglei5205
"""
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

### 载入数据
print('载入数据')
df_train = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data1.csv')
df_test = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data2.csv')

y_train = df_train['is_trade']
y_test = df_test['is_trade']
X_train=df_train.drop(['is_trade','instance_id],axis=1)
X_test=df_test.drop(['is_trade','instance_id'],axis=1)

### 建立模型
print('# 建立模型')
estimator = LGBMClassifier(
                     objective='binary',            # 二分类问题
                     num_leaves=31,                 # 默认31， 叶子个数
                     learning_rate=0.3,             # 默认0.1，学习率
                     n_estimators=67,               # 默认10，决策树个数
                     subsample_for_bin=1  ,         #
                     subsample=1,                   # 默认1，
                     metric = 'logloss',            # 评估指标
                     silent =True,                  # 输出中间过程
                     reg_alpha=0.0,                 # L1正则化系数
                     min_split_gain=0.0,            # 默认0，分裂最小权重
                     early_stopping_rounds=50       # 提前终止训练
                     )

### 网格搜素
print('# 网格搜索')
param_grid = {
        'num_leaves':list(range(25,80,5)),
        'min_child_weight':list(range(2,10,1))
        
        }

gs = GridSearchCV(estimator,              # 分类器
                  param_grid,             # 参数字典
                  scoring='neg_log_loss', # 评价标准
                  cv=3,                   # 三折交叉验证
                  verbose = 2,            # 打印全部中间过程（0/1/2）
                  n_jobs=1)               # 并行计算CPU个数

gs.fit(X_train,y_train)
print('最佳参数:',gs.best_params_)
print('最优分数:',gs.best_score_)

### 训练模型
print('# 训练模型')
lgbm = gs.best_estimator_                     # 最优分类器
lgbm.fit(X_train,y_train)                     # 模型训练

# 模型属性
print('best_score:',lgbm.best_score)          # 最优分数
print('best_iteration:',lgbm.best_iteration)  # 最佳迭代器个数（早停系数）

### 模型预测
print('# 模型预测')
y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)

### 性能评估
print('log_loss',log_loss(y_test, y_pred))

### 特征选择
#print('特征选择')
#print('打印Feature importances:', list(lgbm.feature_importances_))