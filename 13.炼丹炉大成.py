# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:51:56 2023

@author: Pluto
"""
# 可以通过修改超参数来改变网络训练效果
# 超级参数，也就是训练过程中无法改变的参数
# 1.network_shape
# 2.batch_size
# 3.learning_rate
# 请拉到最底下进行修改

import numpy as np
import creatdataandplot as cp
import copy
import math

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
# 精确的损失函数
def precise_loss_function(predicted,real):
    real_matrix = np.zeros((len(real),2))  # len(real)行，2列
    real_matrix[:,1] = real
    real_matrix[:,0] = 1-real
    product = 1-np.sum(predicted*real_matrix,axis=1)  # 交叉熵
    return product
# 相对宽容的损失函数
def loss_function(predicted,real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition,1,0)  #先四舍五入
    real_matrix = np.zeros((len(real),2))
    real_matrix[:,1] = real
    real_matrix[:,0] = 1-real
    product = 1-np.sum(binary_predicted*real_matrix,axis=1)
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
    # 单批次训练
    def one_batch_train(self,batch):
        global force_train,random_train,n_improved,n_not_improved
        inputs = batch[:,(0,1)]
        targets = copy.deepcopy(batch[:,2]).astype(int)  # 标准答案
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], targets)
        loss = loss_function(outputs[-1], targets)
        
        if np.mean(precise_loss) <= 0.1:  # 这个是猜的，也可以所有情况都要训练
            print("No need for training")  # 太准确了就不用了
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], targets)
            backup_loss = loss_function(backup_outputs[-1], targets)
            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):  # 比较新老网络的损失函数
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()  # copy保证前后互不干扰
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print("Improved")
                n_improved += 1
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print("Force train!")
                if random_train:
                    self.random_update()
                    print("Random update!")
                else:
                    print("NO improvement")
                n_not_improved += 1
        print("-"*30)


    # 多批次训练
    def train(self,n_entries):
        global force_train,random_train,n_improved,n_not_improved
        n_improved = 0
        n_not_improved = 0
        
        n_batches = math.ceil(n_entries/BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(100)  # 这里的值需要注意一下
            self.one_batch_train(batch)

        improvement_rate = n_improved/(n_improved+n_not_improved)
        print("Improvement rate:{:%}%.".format(improvement_rate))
        if improvement_rate <= 0.05:  # 可更改
            force_train = True
        else:
            force_train = False
        if n_improved == 0:
            random_train = True
        else:
            random_train = False

        data = cp.creat_data(BATCH_SIZE)
        inputs = data[:,(0,1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:,2] = classification
        cp.plot_data(data, "After training")
        
    # 随机更新
    def random_update(self):
        random_network = network(network_shape)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weights_change
            self.layers[i].biases += biases_change

def main():
    global force_train,random_train,n_improved,n_not_improved
    n_improved = 0
    n_not_improved = 0

    data = cp.creat_data(BATCH_SIZE)
    cp.plot_data(data,"Right classification")
    # 选择起始网络,一般来讲蓝黄相间的好一些。
    use_this_network = 'n'
    while use_this_network != 'y' and use_this_network != 'Y':
        network1 = network(network_shape)
        inputs = data[:,(0,1)]
        outputs = network1.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:,2] = classification
        cp.plot_data(data, "Choose network")
        use_this_network = input('Use this network?(y\n)\n')
        
    # 训练循环，注意，一旦网络死掉了就可以拜拜了
    do_train = input('Train?(y\n)\n')
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric() == True:
            n_entries = int(do_train)
        else:
            n_entries = int(input('Enter the number of data entries used to train:\n'))
        network1.train(n_entries)
        do_train = input('Train?(y\n)\n')
        
    # 演示训练效果
    inputs = data[:,(0,1)]  # 去掉打过标签的数据
    outputs = network1.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:,2] = classification
    cp.plot_data(data, "After training")
    

## 超级参数们
network_shape = [2,30,40,50,2]  # 修改请注意，第一层和最后一层要和input\output匹配
BATCH_SIZE = 100  #同时影响生成数据的个数
learning_rate = 0.02
force_train = False  # 强制训练，在梯度消失的时候，不再改进的时候强制更新,在多批次训练函数中修改触发条件
random_train = False  # 强制训练都不管用了，随机搞一搞吧。让网络起死回生
n_improved = 0  #优化的次数，也就是把Improved和No improved计数，用于前面两个起死回生措施的依据
n_not_improved = 0
## 正式开始
main()