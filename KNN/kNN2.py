# -*- coding:utf-8 -*-
"""
@author:Tian Sir
@file:kNN2.py
@time:2018/3/2121:48
"""

from numpy import *
import operator

def createDataSet():                                                     #创建一个分类的目标（目标，和标号）
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]      #返回数组的行数  shape为求数组的行和列 shape[0]为行，shape[1]为列
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet    #前面的 tile可以实现将 inX 变成 与 dataSet行数相同的
                                                       # （tile有复制的功能，dataSetSize保证有复制的次数，‘1’为每行的个数）
                                                        #再进行与 dataSet 相减
    sqDiffMat = diffMat ** 2                           #对 上一行的到 相减的结果求平方
    sqDistances = sqDiffMat.sum(axis=1)                #用上一行得到 x y 坐标的各平方 ， sqDiffMat.sum(axis=1)实现
                                                       # 将两者相加 即 （X1 - X2）**2 + (Y1 - Y2) ** 2
    distances = sqDistances ** 0.5                     #对上边开方求得距离
    sorteDistIndicies = distances.argsort()            #将上边的距离进行由小到大排序，返回大小的小标例如 [2,0,1]---》[1,0,1]
    classCount = {}                                    #定义一个分类存储的字典
    for i in range(k):
        voteIlabel = labels[sorteDistIndicies[i]]      #返回列表
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #get为取字典中的元素，如果之前的voteIlabel的值是有的，
                                                                     # 那么将返回之前的值，此为计算K个距离最小的点的类别，这个点为那个类别，那个类别就加1
        print('i', i, '\n', 'sorteDistIndicies[i]',sorteDistIndicies[i], '\n','voteIlabel', voteIlabel, '\n','classCount',classCount,'\n',
              'classCount[voteIlabel]',classCount[voteIlabel])

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),
                              reverse=True)   #operator.itemgetter,实现对字典中的第一个排序，itemgetter(1)为对字典的值排序，‘AB’为第零个，即对1 2 进行按降序排列

    return sortedClassCount[0][0], sorteDistIndicies, voteIlabel,classCount[voteIlabel], sortedClassCount, distances

group, labels = createDataSet()
a,b,c, d,e,f = classify0([0,0], group, labels, 4)

print(a,'\n',b,'\n',c,'\n',d,'\n',e,'\n',f)


