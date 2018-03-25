# -*- coding: utf-8 -*-
"""
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 博客：http://cnblogs.com/wanglei5205
# github：http://github.com/wanglei5205
"""

# 真实值和预测值
import numpy as np
y_test = np.array([0,0,1,1])
y_pred1 = np.array([0.3,0.2,0.25,0.7])
y_pred2 = np.array([0,0,1,0])

### 性能度量auc
from sklearn.metrics import roc_auc_score

# 预测值是概率
auc_score1 = roc_auc_score(y_test,y_pred1)
print(auc_score1)

# 预测值是类别
auc_score2 = roc_auc_score(y_test,y_pred2)
print(auc_score2)