# coding:utf-8
import numpy as np # 科学计算库
import collections # 集合模块

"""
# 实现KNN算法，并使用算法分类
"""

"""
### kNN分类器
# 参数:
    dataSet 特征（训练集）
    labels  标签（训练集）
    k -     kNN算法参数,选择距离最小的k个点
    inX -   特征（测试集）
    
# 返回值:
    sortedClassCount[0][0] - 分类结果（标签）
"""
def classify0(dataset,labels,k,inx):
    
    # 计算欧氏距离
    dist = np.sum((inx - dataset)**2, axis=1)**0.5
    
    # 距离递增排序--k个最近的标签
    k_labels = [labels[index] for index in dist.argsort()[0:k]]

    # 统计标签个数--出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]

    return label

"""
### 测试程序
# 创建训练集和测试集，使用KNN分类，打印分类结果
"""

if __name__ == '__main__':
    
    # 创建数据集
    # 训练集
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    
    print("# 特征\n",group)
    print("# 标签\n",labels)
          
    # 测试集
    test = [101,20]

    # kNN分类
    test_class = classify0(group,labels,3,test)
    
    # 打印分类结果
    print(test_class)