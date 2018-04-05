#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:19:09 2018

@author: hello4720
"""

import lightgbm as lgb 
#from pylightgbm.models import GBMRegressor
import time
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd


print("Loading Data ... ")

#读取数据
dataset1 = pd.read_csv('./data/7_train_data1.csv')
dataset2 = pd.read_csv('./data/7_train_data2.csv')
dataset3 = pd.read_csv('./data/7_train_data3.csv')
dataset4 = pd.read_csv('./data/7_train_data4.csv')
dataset5 = pd.read_csv('./data/7_train_data5.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)
dataset4.drop_duplicates(inplace=True)
dataset5.drop_duplicates(inplace=True)

#合并数据集
trains = pd.concat([dataset1,dataset2],axis=0)
trains = pd.concat([trains,dataset3],axis=0)
trains = pd.concat([trains,dataset4],axis=0)

online_test = dataset5

print("loaded data!")

#拆分数据集
train_xy,offline_test = train_test_split(trains, test_size = 0.2,random_state=21)
train,val = train_test_split(train_xy, test_size = 0.2,random_state=21)

#ks评价函数
def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()

y = train.is_trade
X = train.drop(['instance_id','is_trade'],axis=1)

val_y = val.is_trade
val_X = val.drop(['instance_id','is_trade'],axis=1)

offline_test_X=offline_test.drop(['instance_id','is_trade'],axis=1)
online_test_X=online_test.drop(['instance_id'],axis=1)

start_time = time.time()
# create dataset for lightgbm
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(X, y, free_raw_data=False)
lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train,free_raw_data=False)

seed=13
params2 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 40,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 1,
    'bagging_freq': 10,
    'verbose': 0,
    #'num_iterations':500,
    'tree_learner':'serial',  
    'min_data_in_leaf':10,
    'feature_fraction_seed':seed,
    'bagging_seed':seed,
    'metric_freq':1,
    'boosting': 'dart',
    'max_bin': 100,
    'lambda_l1': 1,
    'lambda_l2': 0,
    #'min_split_gain': 0.1
}

print('Start training...')
# train
gbm = lgb.train(params2,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Save model...')
# save model to file
gbm.save_model('./model/lgb_03_31.model') # 用于存储训练出的模型

print ("跑到这里了model.predict")
preds_offline = gbm.predict(offline_test_X, num_iteration=gbm.best_iteration)#
preds_online = gbm.predict(online_test_X, num_iteration=gbm.best_iteration)#
offline=offline_test[['instance_id','is_trade']]
online=online_test[['instance_id']]

offline['preds']=preds_offline
online['preds']=preds_online

print("线下得分;")
#print(ks(offline.preds,offline.is_trade))
offline.is_trade = offline['is_trade'].astype(np.float64)
print('log_loss', log_loss(offline.is_trade, offline.preds))

print ("跑到这里了,输出结果")
#from sklearn.preprocessing import MinMaxScaler
#online.preds = MinMaxScaler().fit_transform(online.preds)
online.rename(columns={'preds':'predicted_score'},inplace=True)
online.to_csv("./submit/20180331.txt",index=None,sep=' ')

#save feature score and feature information:  feature,score,min,max,n_null,n_gt1w
print('Calculate feature importances...')
#print('Feature importances:', list(gbm.feature_importance()))
#print('Feature importances:', list(gbm.feature_importance("gain")))
df = pd.DataFrame(X.columns.tolist(), columns=['feature'])
df['importance']=list(gbm.feature_importance())
df = df.sort_values(by='importance',ascending=False)
df.to_csv("feature_score_20180331.csv",index=None,encoding='gbk')
df











