# -*- coding: utf-8 -*-
"""
# 时间：2018.3.25
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 博客：http://cnblogs.com/wanglei5205
# github：http://github.com/wanglei5205
"""
### 导入模块
import pandas as pd                                # 数据分析库
import numpy as np                                 # 科学计算库
import matplotlib.pylab as plt                     # 数据可视化库
import xgboost as xgb                              # xgboost模型
from xgboost.sklearn import XGBClassifier          # xgboost模型（sklearn）
from sklearn import cross_validation               # 交叉验证
from sklearn import metrics                        # 性能度量
from sklearn.model_selection import GridSearchCV   # 网格搜索

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

### 载入数据
train = pd.read_csv("G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data1.csv")  # 有标签的训练集
test = pd.read_csv("G:/ML/ML_match/IJCAI/data3.22/3.22ICJAI/data/7_train_data5.csv")   # 无标签的测试集
#test_results = pd.read_csv('test_results.csv')                                         # 测试集标签

# 统一标签
target = 'is_trade'
IDcol = 'instance_id'
predictors = [x for x in train.columns if x not in [target, IDcol]]  # 特征列表

### 训练模型（交叉验证）
"""
# alg                    算法
# dtrain                 训练集
# dtest                  测试集 
# predictors  
# useTrainCV 
# cv_flods 
# early_stopping_rounds 

# XGBClassifier是封装在sklearn中的xgboost模型
"""
def modelfit(alg, dtrain, dtest, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    # xgboost.cv()交叉验证
    if useTrainCV:
        # 获取XGBClassifier的参数
        xgb_param = alg.get_xgb_params()
        
        # 训练集和测试集转换为xgboost
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        # xgtest = xgb.DMatrix(dtest[predictors].values)
        
        # xgboost.cv()交叉验证
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          metrics='logloss',
                          early_stopping_rounds=early_stopping_rounds, 
                          show_stdv=False)
        
        # 设置XGBClassifier参数
        alg.set_params(n_estimators=cvresult.shape[0])
    
    # 训练XGBClassifier
    alg.fit(dtrain[predictors],dtrain[target],eval_metric='logloss')
        
    # 模型预测
    dtrain_predictions = alg.predict(dtrain[predictors])         # 返回类别
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1] # 返回概率
        
    # 性能评估
    print("\nModel Report")
    print("logloss : %.6g" % metrics.log_loss(dtrain[target].values, dtrain_predictions))  # 负对数似然损失
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)) # AUC值
    
    # 知道测试集target时使用
    #dtest['predprob'] = alg.predict_proba(dtest[predictors])[:,1]
    #results = test_results.merge(dtest[[IDcol,'predprob']], on='ID')
    #print('AUC Score (Test): %f' % metrics.roc_auc_score(results[target], results['predprob']))
    
    # 
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')


### 建立模型
print('### 建立XGBClassifier')
xgbc = XGBClassifier(learning_rate =0.1,                 # 学习率
                     silent = 1,                         # 输出中间过程
                     n_estimators=150,                   # 决策树个数
                     max_depth=5,                        # 决策树深度
                     min_child_weight=1,                 # 最小叶子节点权重和？
                     gamma=0,                            # 惩罚项系数
                     subsample=0.8,                      # 训练一棵树所用的数据占全部数据集比例
                     colsample_bytree=0.8,               # 训练一颗树所用的特征占全部数据集比例
                     objective= 'binary:logistic',       # 损失函数 
                     nthread=4,                          # 线程数
                     scale_pos_weight=1,                 # 样本不平衡
                     eval_metric='logloss',              # 评估指标
                     reg_alpha=0.03,                     # 正则化系数
                     seed=27)                            # 随机种子

### 网格搜索
## step1:决策树个数 n_estimators
print("### 调参:决策树个数")
#modelfit(xgbc, train, test, predictors)

## step2:决策树参数 max_depth/min_child_weight/gamma/subsample/colsample_bytree
print("### 调参:决策树参数")
param_test1 = {'max_depth':list(range(3,10,2)),'min_child_weight':list(range(1,6,2))}
param_test2 = {'max_depth':[4,5,6],'min_child_weight':[4,5,6]}
param_test2b ={'min_child_weight':[6,8,10,12]}
param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
param_test4 = {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}
param_test5 = {'subsample':[i/100.0 for i in range(75,90,5)],'colsample_bytree':[i/100.0 for i in range(75,90,5)]}

## step3:正则化参数 reg_alpha
print("### 调参：正则化参数")
param_test6 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
param_test7 = {'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]}

gsearch = GridSearchCV(estimator = xgbc,         # 待调参模型
                       param_grid = param_test1, # 参数字典
                       scoring='neg_log_loss',   # 评价函数
                       n_jobs=1,                 # 并行CPU个数
                       iid=False,                # 独立同分布
                       verbose=2,                # 显示中间过程
                       cv=5)                     # 5折交叉验证
                       
gsearch.fit(train[predictors],train[target])     # 训练GridSearchCV

print(gsearch.best_params_)                      # GridSearchCV最佳参数
print(gsearch.best_score_)                       # GridSearchCV最佳分数

xgbc = gsearch.best_estimator_                   # GridSEarchCV最佳分类器

### 训练XGBClassifier模型
print("### 训练XGBClassifier")
modelfit(xgbc, train, test, predictors)