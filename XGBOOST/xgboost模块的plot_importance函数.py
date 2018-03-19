# -*- coding: utf-8 -*-
"""
###############################################################################
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 代码：http://github.com/wanglei5205
# 博客：http://cnblogs.com/wanglei5205
# 目的：学习xgboost的plot_importance函数
###############################################################################
"""
### load module
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits              # 载入数据
from sklearn.model_selection import train_test_split  # 数据分割
from xgboost import XGBClassifier                     # 载入模型
from xgboost import plot_importance                   # 特征权重
from sklearn.metrics import accuracy_score            # 模型评估

### load datasets
digits = load_digits()                                # 载入mnist数据集

### data analysis
print(digits.data.shape)                              # 打印输入空间维度
print(digits.target.shape)                            # 打印输出空间维度

### data split 
x_train,x_test,y_train,y_test = train_test_split(digits.data,      
                                                 digits.target,
                                                 test_size = 0.3,  # 测试集占30%
                                                 random_state = 33)# 随机种子
### fit model for train set 
model = XGBClassifier()
model.fit(x_train,y_train)

### make prediction for test data
y_pred = model.predict(x_test)

### model evaluate 
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

### plot feature importance
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

"""
95.0%
"""