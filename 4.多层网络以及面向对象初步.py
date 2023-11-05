# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:07:39 2023

@author: Pluto
"""
import numpy as np

# 权重生成函数,接收n个输入，m个神经元
def Generate_Weights(n_inputs,n_neurons):
    return np.random.randn(n_inputs,n_neurons)
# 偏置值生成函数
def Generate_bias(n_neurons):
    return np.random.randn(n_neurons)
# ReLu激活函数
def ReLu(inputs):
    return np.maximum(0,inputs)

#网络图示
#         o
#     o       
#  o      o   o
#     o       
#  o      o   o
#     o       
#         o

# array.a
a11,a21=0.9,-0.4
a12,a22=-0.8,0.5
a13,a23=-0.5,0.8
a14,a24=0.7,-0.3
a15,a25=-0.9,0.4

#batch
inputs = np.array([[a11,a21],
                   [a12,a22],
                   [a13,a23],
                   [a14,a24],
                   [a15,a25]]
                  )
#第一层
weights1 = Generate_Weights(2, 3)
biases1 = Generate_bias(3)
sum1 = np.dot(inputs, weights1) + biases1
output1 = ReLu(sum1)
## print(output1)
## print(5*"-"+"第1层结束"+5*"-")

#第二层
weights2 = Generate_Weights(3, 4)
biases2 = Generate_bias(4)
sum2 = np.dot(output1, weights2) + biases2
output2 = ReLu(sum2)
## print(output2)
## print(5*"-"+"第2层结束"+5*"-")

#第三层
weights3 = Generate_Weights(4, 2)
biases3 = Generate_bias(2)
sum3 = np.dot(output2, weights3) + biases3
output3 = ReLu(sum3)
## print(output3)
## print(5*"-"+"第3层结束"+5*"-")

# 进一步简化
class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)
        
    def layer_forward(self,inputs):
        sum = np.dot(inputs,self.weights) + self.biases
        self.output = ReLu(sum)
        return self.output


inputs = np.array([[a11,a21],
                   [a12,a22],
                   [a13,a23],
                   [a14,a24],
                   [a15,a25]]
                  )

layer1 = layer(2,3)
layer2 = layer(3,4)
layer3 = layer(4,2)

output1 = layer1.layer_forward(inputs)
output2 = layer2.layer_forward(output1)
output3 = layer3.layer_forward(output2)

# 甚至可以：
output3=layer3.layer_forward(layer2.layer_forward(layer1.layer_forward(inputs)))

print(output1)
print(output2)
print(output3)