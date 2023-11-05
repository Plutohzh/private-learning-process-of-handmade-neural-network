# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:52:23 2023

@author: Pluto
"""
import numpy as np

# a1a2a3用w11w21w31进入神经元c
# a1a2a3用w12w22w32进入神经元c'
a1,a2,a3 = -0.9,-0.5,-0.7

w11,w21,w31 = 0.8,-0.4,0
w12,w22,w32 = 0.7,-0.6,0.2

array_a = np.array([a1,a2,a3])
mat_w = np.array([[w11,w12],
                   [w21,w22],
                   [w31,w32]])
b1 = 0.6
def ReLu(inputs):
    return np.maximum(0,inputs)
sum1 = np.dot(array_a, mat_w) + b1
print(sum1)
print("-"*10)
print(ReLu(sum1))

# 我们现在只输入了一个数据a
# 我们还需要batch，也就是一批数据
a1,a2,a3 = -0.9,-0.5,-0.7
b1,b2,b3 = -0.8,-0.5,-0.6
c1,c2,c3 = -0.5,-0.8,-0.2

mat_batch = np.array([[a1,a2,a3],
                      [b1,b2,b3],
                      [c1,c2,c3]])

sum2 = np.dot(mat_batch,mat_w) + b1
print("-"*20)
print(sum2)
print("-"*10)
print(ReLu(sum2))
# 以上完成了一层两个神经元的网络

## 规模化方法
# 权重生成函数,接收n个输入，m个神经元
def Generate_Weights(n_inputs,n_neurons):
    # numpy自带的随机生成标准正态分布中的数
    # 两个参数代表矩阵是n行m列
    return np.random.randn(n_inputs,n_neurons)
    
mat_w=Generate_Weights(3, 5)

# 把偏置值改一改，m个神经元最好有m个偏置值
b1 = np.array([0.5,0.6,0.7,0.8,0.9])

# numpy有一个特性，A=([1,2],[3,4])加B=([-1,-2])
# 得到的结果是([0,0],[2,2]),这在线代中是不允许的
# 为了保证偏置值没问题，我们不妨也引入函数

def Generate_bias(n_neurons):
    return np.random.randn(n_neurons)

b1 = Generate_bias(5)

sum3 = np.dot(mat_batch,mat_w) + b1

print("-"*20)
print(sum3)
print("-"*10)
print(ReLu(sum3))


