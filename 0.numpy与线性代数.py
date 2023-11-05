# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:34:14 2023

@author: Pluto
"""

import numpy as np

#------------------------------------------------------------------------------
## 数组的建立
array0 = np.zeros((2,3))  # 建立一个数组，数组两行三列，所有元素都是零
array1 = np.full((2,3),1)  # 建立一个两行三列，元素全都是1的数组。

print(array0)
print(array1)

# 也可以将列表转化为数组
list1=[[1,2],
       [2,3],
       [5,6]]
array2 = np.array(list1)

print(array2)

#------------------------------------------------------------------------------
## 数组的优势
# np库比较强大的是：看我索引
index = array2[:,0]  # 第一列的所有行
index2 = [list1[0][0],list1[1][0],list1[2][0]]
index3 = array2[1,:]  # 第二行的所有列
print(index)
print(index2)

#------------------------------------------------------------------------------
## 线性代数的运算
a = np.array([1,2,3])
b = np.array([3,4,5])
c = np.array([[1,4],
             [2,5],
             [3,6]])
d = np.array([[1,2,3],
              [4,5,6]])
e = np.array([[-1,2],
              [3,-4]])
f = np.array([[0,2],
              [3,1]])


x = np.dot(a,b)  # 矩阵点乘
print(x)
y = e * f  # 矩阵花乘，CAST，也就是每个位置对应乘
print(y)
print(x+y)  # 矩阵相加，这里x=26,x+y时将x作为arrayfull(2x2,26)

#------------------------------------------------------------------------------
##求统计量
# 平均数
av_array = np.array([[1,2],
                [3,4],
                [5,6]])
av1 = np.mean(av_array,axis=0)  # 每列求平均，得到一个列表，其实是纵轴
av2 = np.mean(av_array,axis=1)  # 每行求平均，得到一个列表，其实是横轴
av3 = np.mean(av_array)  # 求所有数的平均

# 最大值
max_array1 = np.array([[-1,2],
                [3,-4],
                [-5,6]])
max1 = np.maximum(0,max_array1)  # 在0和a之间取最大。如果矩阵中某个数a小于0则取0，若大于0则取a
'''得到[[0 2]
        [3 0]
        [0 6]]
'''
print(max1)

max_array2 = np.array([[0,1],
                [2,-5],
                [3,0]])
max2 = np.maximum(max_array2,max_array1)  # 两个矩阵对应位置最大值
'''得到[[0 2]
        [3 -4]
        [3 6]]
'''
print(max2)

try:
    max3 = np.max(max2)  # 取某个矩阵所有元素的最大值
    max4 = np.max(max2,axis=1)  # 同上，可以取某个轴
    print(max3)
    print(max4)
except:
    print("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all")


