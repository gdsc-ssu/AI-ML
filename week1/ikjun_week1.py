# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:36:15 2021

@author: 손익준
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:40:05 2021

@author: 손익준
"""

from sklearn.datasets import load_boston

import numpy as np

boston = load_boston()

train_size = 450


train_data = np.array(boston.data[:train_size])
train_value = np.array( boston.target[:train_size])



check_data =np.array( boston.data[train_size:])
check_value = np.array(boston.target[train_size:])




W = np.random.normal(0,1,(13,))

LR = 1.0e-8

for NN in range(100000):
    
    for i in range(train_size):
        
        x = train_data[i]
        y = train_value[i]
        
        f = (x*W).sum() /13
        #print("**f** : ", f )  # f(x)
        #print("**y** : ", y )  # y
        
        cost = ( y - f)**(2)
        #print("**cost** : ",cost)  # cost()
        
        
        #print( "W", W )
        
        #W = W - W * 2*(f-y) * LR  # W update
        W = W - W * 2*(f-y) * LR  # W update
    
    
    
    cost_sum=0
    for i in range(505 - train_size):
        
        
        x = check_data[i]
        y = check_value[i]
        
        
        
        f = (x*W).sum() /13
        #print("**f** : ", f )  # f(x)
        #print("**y** : ", y )  # y
        
        cost = ( y - f)**(2)
        
        
        cost_sum += cost
    if(NN%1000 ==0):
        print("**cost** : ",cost/55)
        rann = np.random.randint(0,50)
        print("**y** :",check_value[rann])
        x = check_data[rann]
        f = (x*W).sum() /13
        print("**predict"" :",f)
        print("--------------")