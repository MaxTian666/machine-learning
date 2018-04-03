# -*- coding:utf-8 -*-
"""
@author:Tian Sir
@file:kNN3.py
@time:2018/3/2214:16
"""
'''
实施KNN算法的步骤：
1）计算已知分类数据集点与当前点的距离；
2）安装距离递增次序排序
3）选取与当前点距离最小的k个点；
4)确定前k个点出现的频率；
5）返回前k个点出现频率最高的类别作为当前点的预分类。
'''
from numpy import *
import operator

'''
创建一个函数用于生成一个待分类的数据及对应的数据标签
'''
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
k-临近算法的实施
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]           #得到已知分类目标的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)     #各平方项求和；即将矩阵的每一行元素相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #实现对矩阵中元素由小到大排序，返回排序后的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #根据sortedDistIndicies中的元素取出 k 个对应的求出labels中存的元素
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #累积求出各标号的个数
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

group, labels = createDataSet()

a = classify0([0,0.1],group,labels, 3)

print(a)