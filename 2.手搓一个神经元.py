# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:24:03 2023

@author: Pluto
"""
import numpy as np

# 输入向量a=(a1,a2,a3),加权w，加上偏置值b进入神经元，利用激活函数判断是否输出

a1 = 0.9
a2 = 0.5
a3 = 0.7

array_a = np.array([a1,a2,a3])

w1 = 0.8
w2 = -0.4
w3 = 0

array_w = np.array([w1,w2,w3])

b1 = 1

sum1 = a1*w1 + a2*w2 + a3*w3 + b1
sum2 = np.dot(array_a,array_w) + b1

# 常用的激活函数：
# Sigmoid函数：输入为R，输出为（0，1）
# ReLu函数：0以下为0，0以上为y=x
def ReLu(inputs):
    return np.maximum(0,inputs)

print(ReLu(sum1))
print(ReLu(sum2))

# 一个神经元至此完成








