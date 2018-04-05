# -*- coding: utf-8 -*-
"""
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 博客：http://cnblogs.com/wanglei5205
# github：http://github.com/wanglei5205
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

# 载入数据
print('载入数据')
dataset1 = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data1.csv')
dataset2 = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data2.csv')
dataset3 = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data3.csv')
dataset4 = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data4.csv')
dataset5 = pd.read_csv('G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data5.csv')

print('删除重复数据')
dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)
dataset4.drop_duplicates(inplace=True)
dataset5.drop_duplicates(inplace=True)

print('数据合并')
trains = pd.concat([dataset1,dataset2],axis=0)
trains = pd.concat([trains,dataset3],axis=0)
trains = pd.concat([trains,dataset4],axis=0)

online_test = dataset5

# 数据拆分(训练集+验证集+测试集)
print('数据拆分')
from sklearn.model_selection import train_test_split
train_xy,offline_test = train_test_split(trains, test_size = 0.2,random_state=21) # 训练集和测试集
train,val = train_test_split(train_xy, test_size = 0.2,random_state=21)           # 训练集和验证集

y = train.is_trade                                                  # 训练集标签
X = train.drop(['instance_id','is_trade'],axis=1)                   # 训练集特征矩阵

val_y = val.is_trade                                                # 验证集标签
val_X = val.drop(['instance_id','is_trade'],axis=1)                 # 验证集特征矩阵

offline_test_X=offline_test.drop(['instance_id','is_trade'],axis=1) # 线下测试特征矩阵
online_test_X=online_test.drop(['instance_id'],axis=1)              # 线上测试特征矩阵

# 建立模型
print('建立模型')
gbm = LGBMClassifier(
                     objective='binary',
                     n_estimators=1000,
                     subsample=0.8,
                     subsample_freq=10,
                     colsample_bytree=0.8,
                     learning_rate = 0.01,
                     reg_alpha=0.8
                     
                     )

# 训练模型
print('训练模型')
gbm.fit(X,                          # 训练集--特征矩阵
        y,                          # 训练集--标签
        eval_set=[(val_X, val_y)],  # 验证集
        eval_metric='logloss',      # 评估标准
        early_stopping_rounds=50,   # 早停系数
        verbose = 2)                # 显示中间过程（0/1/2）

print('best_score:',gbm.best_score)
print('best_iteration:',gbm.best_iteration)

# 线下测评
print('线下预测')
preds_offline = gbm.predict(offline_test_X, num_iteration=gbm.best_iteration) # 线下
offline=offline_test[['instance_id','is_trade']]
offline['preds']=preds_offline

offline.is_trade = offline['is_trade'].astype(np.float64)
print('线下分数log_loss', log_loss(offline.is_trade, offline.preds))

# 线上测评
print('线上预测')
preds_online = gbm.predict(online_test_X,num_iteration=gbm.best_iteration)    # 线上
online=online_test[['instance_id']]
online['preds']=preds_online

print('保存结果')
online.rename(columns={'preds':'predicted_score'},inplace=True)
online.to_csv("G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/20180331.txt",index=None,sep=' ')

# 特征选择
df = pd.DataFrame(X.columns.tolist(), columns=['feature'])
df['importance']=list(gbm.feature_importances_)
df = df.sort_values(by='importance',ascending=False)
df.to_csv("G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/feature_score_20180331.csv",index=None,encoding='gbk')