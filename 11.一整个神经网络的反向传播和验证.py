# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:41:47 2023

@author: Pluto
"""

import numpy as np
import creatdataandplot as cp
import copy

##消除被除数为零的警告
np.seterr(divide='ignore', invalid='ignore')

##标准化函数,[2,10]->[0.2,1]
# 一般情况
def normalize(array):
    max_number = np.max(np.absolute(array),axis=1,keepdims=True)  # 找到绝对值最大的，例如[5,-6]->[5/6,-1]
    scale_rate = np.where(max_number == 0,1,1/max_number)  # 如果最大值为0则为1，如[0,0]->[0,0]若不为零则等于1/max_number
    norm = array * scale_rate  # 标准化
    return norm
# 单个向量的标准化
def vector_normalize(array):
    max_number = np.max(np.absolute(array))  # 找到绝对值最大的，例如[5,-6]->[5/6,-1]
    scale_rate = np.where(max_number == 0,1,1/max_number)  # 如果最大值为0则为1，如[0,0]->[0,0]若不为零则等于1/max_number
    norm = array * scale_rate  # 标准化
    return norm


##softmax后是概率，不能直接打标签
#分类函数
def classify(probabilities):
    classification = np.rint(probabilities[:,1])  # 只取所有行，第二列;rint为四舍五入
    return classification
    
##激活函数
# ReLu激活函数，不适合用在最后一层
def ReLu(inputs):
    return np.maximum(0,inputs)
# softmax激活函数，结果是出现的概率
def Softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)  # 每一行选取最大值组成一列，keepdims保持维度，也就是一列而非一行
    slided_inputs = inputs - max_values  # inputs最大值变成0，叫做滑动，在指数函数上，x1x2等距滑动，y2/y1不变，是为了防止指数爆炸
    exp_values = np.exp(slided_inputs)  # 原本是0的变成1，小于零的（被滑动到负值）变成大于零小于一的数，把区间锁定在(0,1]
    norm_base = np.sum(exp_values,axis=1,keepdims=True)  # 标准化的底
    norm_value = exp_values/norm_base  # 标准化的结果，等比缩小，使得每一行和为1
    return norm_value

##损失函数
# 使用交叉熵比较两个向量的差值(cross entropy)
# [a,b]交叉熵[c,d]=ac+bd
def precise_loss_function(predicted,real):
    real_matrix = np.zeros((len(real),2))  # len(real)行，2列
    real_matrix[:,1] = real
    real_matrix[:,0] = 1-real
    product = 1-np.sum(predicted*real_matrix,axis=1)  # 交叉熵
    return product

##需求函数
# 反向传播算法的起点，确定最后一层
# 确定sum(a1*w1)+b=z,z应当如何调整
# 例如
def get_final_layer_preact_demands(predicted_values,target_vector):
    target = np.zeros((len(target_vector),2))
    target[:,1] = target_vector
    target[:,0] = 1-target_vector
    for i in range(len(target_vector)):
        if np.dot(target[i],predicted_values[i]) > 0.5:
            target[i] = np.array([0,0])  # 无需修改
        else:
            target[i] = (target[i] - 0.5) * 2  # 得到需求函数
    return target

##


## 神经网络的一个层
class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)

    # 前向传播        
    def layer_forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        return self.output

    # 反向传播
    def layer_backward(self,preweights_values,afterweights_demands):
        preweights_demands = np.dot(afterweights_demands,self.weights.T)
        
        condition = (preweights_values > 0)
        value_derivatives = np.where(condition,1,0)  # 求导
        preacts_demands = value_derivatives * preweights_demands
        norm_preacts_demands = normalize(preacts_demands)
        
        weight_adjust_matrix = self.get_weight_adjust_matrix(preweights_values,afterweights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)
        
        return (norm_preacts_demands,norm_weight_adjust_matrix)


    def get_weight_adjust_matrix(self,preweights_values,postweights_demands):
        plain_weights = np.full(self.weights.shape,1)  # 生成一个全都是1的矩阵
        weights_adjust_matrix = np.full(self.weights.shape,0.0)  #后面保证是浮点数
        plain_weights_T = plain_weights.T
        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T*preweights_values[i,:]).T*postweights_demands[i,:]
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix

## 一个神经网络
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
                layer_output = Softmax(layer_sum)  # 最后一次不ReLu,用softmax
            outputs.append(layer_output)
        return outputs
    # 反向传播函数
    def network_backward(self,layer_outputs,target_vector):
        backup_network = copy.deepcopy(self)  # 备用，因为更新之后不一定更准确
        preact_demands = get_final_layer_preact_demands(layer_outputs[-1],target_vector)
        # 每一层都传播一下
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers)-(1+i)]  # 修改备份的网络，并且是倒序
            # 修正最后一层以外的biases
            if i != 0:
                # 引入学习率，也就是步子大了容易跳过最低点，步子太小效率太低
                layer.biases += learning_rate * np.mean(preact_demands,axis=0)
                layer.biases = vector_normalize(layer.biases)
            outputs = layer_outputs[len(layer_outputs)-(2+i)]
            results_list = layer.layer_backward(outputs, preact_demands)
            preact_demands = results_list[0]
            weights_adjust_matrix = results_list[1]
            layer.weights += learning_rate * weights_adjust_matrix  # 更新权重矩阵
            layer.weights = normalize(layer.weights)  # 标准化一下就可以输出了
        
        return backup_network  # 这就是一个更新后的网络了
        
            



## batch,随机生成数据
BATCH_SIZE = 100
data = cp.creat_data(BATCH_SIZE)
learning_rate = 0.01  # 主观性很强，就是猜一个炼丹炉

cp.plot_data(data, "original(right)")
print("原始数据")
print(data)
inputs = data[:,(0,1)]  # 去掉打过标签的数据
targets = copy.deepcopy(data[:,2])  # 标准答案,这里用copy避免data[:,2]变化导致这里的值变化

## 实例
network_shape1 = [2,3,4,5,2]  # 输入层输出层必须和输入输出的维度相同
network1 = network(network_shape1)
output1 = network1.network_forward(inputs)
classification1 = classify(output1[-1])  # 分类完成
data[:,2] = classification1  # 注意，改变data,过去赋值的也会变

print("网络识别的数据")
print(data)  # 输出最后的输出（用softmax标准化后，再分类）
cp.plot_data(data, "before training")


# 反向传播
backup_network1 = network1.network_backward(output1,targets)
# 检测一下
new_outputs = backup_network1.network_forward(inputs)
new_classification = classify(new_outputs[-1])
data[:,2] = new_classification
cp.plot_data(data,"After training1")  # 经过了一次训练，但还是一坨狗屎
