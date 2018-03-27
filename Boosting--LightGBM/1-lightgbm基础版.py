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
gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=100)

# 训练模型
print('训练模型')
gbm.fit(X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

# 模型预测
print('模型预测')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 性能评估
print('log_loss', log_loss(y_test, y_pred) ** 0.5)

# 特征选择
print('特征选择')
print('打印Feature importances:', list(gbm.feature_importances_))