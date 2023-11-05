# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:36:08 2023

@author: Pluto
"""
# 生成数据和可视化

import numpy as np
import random
import matplotlib.pyplot as plt

n_data = 800 # 生成n条数据
classification_type ='ring'  # 默认的图像

def tag_entry(x, y):
    if classification_type == 'circle':
        if x**2+y**2 > 1: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0
    
    if classification_type == 'ring':
        if x**2+y**2 > 1 and x**2+y**2 < 2:  # 圆环
            tag = 1
        else:# 
            tag = 0    
    elif classification_type == 'line':  # y轴
        if x > 0:
            tag = 1
        else:
            tag = 0
            
    elif classification_type == 'cross':
        if x*y > 0:
            tag = 1
        else:
            tag = 0   
    return tag

def creat_data(n_data):
    entry_list = []
    for i in range(n_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry = [x,y,tag]
        entry_list.append(entry)
    return np.array(entry_list)

##可视化
def plot_data(data, title):
    color = []
    for i in data[:, 2]:  # 第三个元素
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")
    
    plt.scatter(data[:, 0], data[:, 1], c=color)  # 散点图
    plt.title(title)
    plt.show()

# 如果程序自己运行，那就出这个，如果是其它调用就不运行
if __name__ == "__main__":
    data = creat_data(n_data)
    print(data)
    plot_data(data, 'Demo')