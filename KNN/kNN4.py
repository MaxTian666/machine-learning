# -*- coding:utf-8 -*-
"""
@author:Tian Sir
@file:kNN4.py
@time:2018/3/2214:59
"""

'''
1）收集数据：提供文本文件
2）准备数据：使用python分析文本数据
3）分析数据：使用Matplotlib画二维扩散图
4）训练算法：不用k-临近算法
5）测试算法：测试样本与非测试样本的区别在于:样本数据已完成分类，如果预测分类与实际分类不同，则标记为一个错误
6）使用算法：产生简单的命令行程序，然后输入一些特征数据进行判断对方是否为自己喜欢的类型
'''
from numpy import *
import operator
import matplotlib
import matplotlib.pylab as plt

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
      #  print('i', i, '\n', 'sorteDistIndicies[i]',sorteDistIndicies[i], '\n','voteIlabel', voteIlabel, '\n','classCount',classCount,'\n',
             # 'classCount[voteIlabel]',classCount[voteIlabel])

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),
                              reverse=True)   #operator.itemgetter,实现对字典中的第一个排序，itemgetter(1)为对字典的值排序，‘AB’为第零个，即对1 2 进行按降序排列

    return sortedClassCount[0][0]
'''
准备数据：从文本文件中解析数据
数据共有三个特征：1）每年获得的飞行唱客里程；2）玩视频游戏消耗时间的百分比；3）每周消耗冰激凌公升数
将上述特征输入到分类器之前，必须将待处理数据的格式改变为分类器可以接受的格式。
创建函数将输入的文本字符串，输出为训练矩阵和标签向量
'''

def file2matrix(filename):
    fr = open(filename)                                         #打开输入的文件
    arrayOLines = fr.readlines()                                #读取文件的多行数据
    numberOfLines = len(arrayOLines)                            #获得多行数据的行数
    returnMat = zeros((numberOfLines, 3))                       #创建返回矩阵行数与文本文件的行数相同，列数为3列
    classLabelVector = []                                       #创建标签向量存储列表
    index = 0
    for line in arrayOLines:
        line = line.strip()                                     #//去除文本文件中的回车符
        listFromLine = line.split('\t')                         #//用tab键将上边得到整数行数据分割为一个列表
        returnMat[index, :] = listFromLine[0:3]                 #选取列表中前三个元素存于矩阵中
        classLabelVector.append(int(listFromLine[-1]))          #将列表中的最后一个元素存于标号向量中
        index += 1                                              #逐行操作
    return returnMat,classLabelVector


def Drawing(datingDataMat1,datingLabels1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat1[:,1],datingDataMat1[:,2])
    #ax1 = fig.add_subplot(121)
    #ax1.scatter(datingDataMat1[:, 1], datingDataMat1[:, 2],15.0*array(datingLabels1),15*datingLabels1)
    plt.show()
'''
归一化数据：
将数据归一化到同一范围
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)                                       #返回矩阵每行的最小值
    maxVals = dataSet.max(0)                                       #返回矩阵每列的最大值
    ranges = maxVals - minVals                                     #计算最大值与最小值之差
    normDataSet = zeros(shape(dataSet))                            #创建一个零矩阵行数与文本数据相同
    m = dataSet.shape[0]                                           #得到dataSet的行数
    normDataSet = dataSet - tile(minVals, (m, 1))                  #利用tile函数实现将最小值复制成与dataSet的行数相同的矩阵，
                                                                   # 并计算两者之差
    normDataSet = normDataSet / tile(ranges, (m, 1))               #将每列的最大值与最小值之差，利用tile函数复制为与dataSet的行数相同，
                                                                   # 并将上边的相减除以此值
    return normDataSet, ranges, minVals

'''
测试分类的正确性
进行评估算法的正确性
'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels =file2matrix('datingTestSet2.txt')  #准备数据 分割特征 及标号
    normMat, ranges, minVals = autoNorm(datingDataMat)              #将数据进行归一化处理
    m = normMat.shape[0]                                            #获得归一化后数据的行数
    numTestVecs = int(m*hoRatio)                                    #计算测试样本的行数 为总样本的 10%
    print(numTestVecs)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)   #计算测试数据中每一行与，剩下90%数据的分类结果
        print('i',i,"the classifier came back with:%d,the real answer is:%d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("The total error rate is: %f"%(errorCount/float(numTestVecs)))
   # print(normMat[numTestVecs:m,:])

'''
使用算法：构建完整可用的系统
构建完整的测试程序
'''

def ClassifyPerson():
    resultList = ['Not at all', 'In small doses', 'In large doses']
    percenTats = float(input("Percentage of time spent playing video games:"))
    ffMiles = float(input("Frequent flier miles earned per year:"))
    iceCream = float(input("Liters of ice cream consumed per year:"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)    #将数据转化为标准的格式（0~1）
    inArr = array([ffMiles, percenTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probably like this person:',resultList[classifierResult - 1])
'''
datingDataMat, datingLabels, classLabelVector = file2matrix('datingTestSet2.txt')
Drawing(datingDataMat,datingLabels)
normMat, ranges, minVals = autoNorm(datingDataMat)



#print('datingDataMat',datingDataMat,'\n','datingLabels',datingLabels,'\n','classLabelVector',classLabelVector)

print('normMat', normMat, '\n', 'ranges',ranges,'\n','minVals',minVals)

'''

#datingClassTest()

ClassifyPerson()

















































































































