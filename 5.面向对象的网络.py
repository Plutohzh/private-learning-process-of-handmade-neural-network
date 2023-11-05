# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:27:21 2023

@author: Pluto
"""
import numpy as np

# ReLu激活函数
def ReLu(inputs):
    return np.maximum(0,inputs)

# 神经网络的一个层
class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)
        
    def layer_forward(self,inputs):
        sum = np.dot(inputs,self.weights) + self.biases
        self.output = ReLu(sum)
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
            layer_output = self.layers[i].layer_forward(outputs[i])
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
outputs1 = network1.network_forward(inputs)

print(network1.shape)
print("-"*20)
print(network1.layers[0].weights)  # 第一层的权重
print("-"*20)
print(network1.layers[1].weights)  # 第二层的权重
print("-"*20)
print(network1.layers[0].biases)  # 第一层的偏置值
print("-"*20)
print(outputs1[-1])  # 最后一层的输出值
print("-"*20)

