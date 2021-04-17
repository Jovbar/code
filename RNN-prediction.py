# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:06:43 2021

@author: Administrator
"""
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

seq_len=50
input_size=1
hidden_size=16
output_size=1
lr=0.01

class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.rnn=nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        for p in self.rnn.parameters():#参数初始化
            nn.init.normal_(p,mean=0.0,std=0.001)
        self.linear=nn.Linear(hidden_size,output_size)
        
    def forward(self, x ,hidden_prev):#h0
        out,hidden_prev=self.rnn(x,hidden_prev)#out,ht=rnn(x,h0)
        out = out.view(-1,hidden_size)#out.shape[batch,seq,hiddensize]=>[seq,hiddensize]
        out = self.linear(out)#[seq,h]=>[seq,1]  output_size=1
        out = out.unsqueeze(dim = 0)#=>[1,seq,1]  because y.shape[batch,seq,1] MSELoss
        return out,hidden_prev
    
model=net()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr)
#构建学习率为lr、优化算法为adam的优化器optimizer，并作用于model参数

##training
hidden_prev=torch.zeros(1,1,hidden_size)#初始化[num_lay,batch_size,hidden_size]的三阶零张量，也可以取随机数
for iter in range(200):
    start = np.random.randint(3,size = 1)[0]#0~3随机初始化
    time_steps = np.linspace(start,start + 10,seq_len)
    data = np.sin(time_steps)#加噪声
    data = data.reshape(seq_len,1)
    x = torch.tensor(data[:-1]).float().view(1,seq_len - 1,1)#0~49
    y = torch.tensor(data[1:]).float().view(1,seq_len - 1,1)#1~50

    output,hidden_prev = model(x,hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output,y)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Iteration: {iter} loss: {loss.item()}")

predictions=[]#input.shape[1,49,1]=>[1,1,1]
input=x[:,0,:]#input.shape[1,seq,1]=>[1,1]/[batchsize,inputsize]
for _ in range(x.shape[1]):
    input=input.view(1,1,1)#input.shape[1,1,1]
    (pred,hidden_prev)=model(input,hidden_prev)#input_seqlen=1
    input=pred#固定测试集
    predictions.append(pred.detach().numpy().ravel()[0])

x=x.data.numpy().ravel()
y=y.data.numpy()
plt.scatter(time_steps[:-1],x.ravel(),s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[:-1],predictions)
plt.show()

 















