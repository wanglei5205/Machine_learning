# -*- coding: utf-8 -*-
"""
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 博客：http://cnblogs.com/wanglei5205
# github：http://github.com/wanglei5205
"""
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

# 载入数据
print('载入数据')
df_train = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data1.csv')
df_test = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data2.csv')

y_train = df_train['is_trade']
y_test = df_test['is_trade']
X_train=df_train.drop(['is_trade'],axis=1)
X_test=df_test.drop(['is_trade'],axis=1)

# 建立模型
print('建立模型')
estimator = lgb.LGBMClassifier(objective='binary',
                               num_leaves=40,
                               learning_rate=1,
                               n_estimators=100,
                               metric='logloss',
                               scale_pos_weight = 50,
                               lambda_l1=1)

# 网格搜素
print('网格搜索')
param_grid = {'learning_rate':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'n_estimators':list(range(20,1500,10))}

gs = GridSearchCV(estimator, param_grid)
gs.fit(X_train, y_train)
print('Best parameters found by grid search are:', gs.best_params_)
print('Best score found by grid search are:',gs.best_score_)

# 训练模型
print('训练模型')
lgbm = gs.best_estimator_
lgbm.fit(X_train,y_train)

# 模型预测
print('模型预测')
y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)

# 性能评估
print('log_loss',log_loss(y_test, y_pred))

# 特征选择
print('特征选择')
print('打印Feature importances:', list(lgbm.feature_importances_))