# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:58:50 2023

@author: Pluto
"""

import numpy as np

##标准化函数,[2,10]->[0.2,1]
def normalize(array):
    max_number = np.max(np.absolute(array),axis=1,keepdims=True)  # 找到绝对值最大的，例如[5,-6]->[5/6,-1]
    scale_rate = np.where(max_number == 0,0,1/max_number)  # 如果最大值为0则为0，如[0,0]->[0,0]若不为零则等于1/max_number
    norm = array * scale_rate  # 标准化
    return norm


##激活函数
# ReLu激活函数，不适合用在最后一层
def ReLu(inputs):
    return np.maximum(0,inputs)
# softmax激活函数
def Softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)  # 每一行选取最大值组成一列，keepdims保持维度，也就是一列而非一行
    slided_inputs = inputs - max_values  # inputs最大值变成0，叫做滑动，在指数函数上，x1x2等距滑动，y2/y1不变，是为了防止指数爆炸
    exp_values = np.exp(slided_inputs)  # 原本是0的变成1，小于零的（被滑动到负值）变成大于零小于一的数，把区间锁定在(0,1]
    norm_base = np.sum(exp_values,axis=1,keepdims=True)  # 标准化的底
    norm_value = exp_values/norm_base  # 标准化的结果，等比缩小，使得每一行和为1
    return norm_value
# 神经网络的一个层
class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)
        
    def layer_forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        return self.output

# 一个神经网络
class network:
    def __init__(self,network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layeri = layer(network_shape[i],network_shape[i+1])
            self.layers.append(layeri)
    def network_forward(self,inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers)-1:  # 如果不是最后一个输出值
                layer_output = normalize(ReLu(layer_sum))  # 每一层都激活+标准化，防止过度发散
            else:
                layer_output = Softmax(layer_sum)
            outputs.append(layer_output)
        return outputs

# batch

a11,a21=0.9,-0.4
a12,a22=-0.8,0.5
a13,a23=-0.5,0.8
a14,a24=0.7,-0.3
a15,a25=-0.9,0.4

inputs = np.array([[a11,a21],
                   [a12,a22],
                   [a13,a23],
                   [a14,a24],
                   [a15,a25]]
                  )

# 实例

network_shape1 = [2,3,4,2]
network1 = network(network_shape1)
output1 = network1.network_forward(inputs)
print(output1[-1])  # 输出最后一个输出值（用softmax标准化后）