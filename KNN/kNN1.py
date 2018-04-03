# -*- coding:utf-8 -*-
"""
@author:Tian Sir
@file:kNN1.py
@time:2018/3/2121:31
"""

from numpy import *
import sys
import operator    #运算符模块

def createDataSet():
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0.1]])      #创建数组
    labels = ['A', 'A', 'B', 'B']                        #定义一个标号
    return group, labels
group, labels = createDataSet()

print('group',group,'\n',type(group),'\n','labels',labels)