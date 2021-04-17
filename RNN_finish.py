# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:53:20 2021

@author: Administrator
"""
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

seq_len=4000
input_size=1
hidden_size=70
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
        #out.shape[batch,seq,hiddensize]=>[seq,hiddensize]
        out = out.view(-1,hidden_size)
        out = self.linear(out)#[seq,h]=>[seq,1]  output_size=1
        out = out.unsqueeze(dim = 0)#=>[1,seq,1]  because y.shape[batch,seq,1] MSELoss
        return out,hidden_prev
    
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

model=net()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr)
#构建学习率为lr、优化算法为adam的优化器optimizer，并作用于model参数

#############################training##########################
hidden_prev=torch.zeros(1,1,hidden_size)#初始化[num_lay,batch_size,hidden_size]的三阶零张量，也可以取随机数
Loss_list = []

time_steps = np.linspace(0,4*np.pi,seq_len)
d1 = np.sin(time_steps)
d0 = np.random.normal(0,0.1,4000)#加噪声
data = d1 + d0
data = data.reshape(seq_len,1)
x_tra = torch.tensor(data[:2999]).float().view(1,seq_len -1001,1)
y_tra = torch.tensor(data[1:3000]).float().view(1,seq_len -1001,1)

for iter in range(201):
    output,hidden_prev = model(x_tra,hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output,y_tra)
    Loss_list.append(loss)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print("Training:" f"Iteration: {iter} loss: {loss.item()}")

plt.plot(time_steps[1:3000], y_tra.flatten(), 'r-')
plt.plot(time_steps[1:3000], output.data.numpy().flatten(), 'b-')
plt.draw(); plt.pause(0.05)

x1 = range(0, 201)
L1 = Loss_list
plt.plot(x1, L1, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
#5181

###########################prediction#############################
hidden_prev=torch.zeros(1,1,hidden_size)#初始化[num_lay,batch_size,hidden_size]的三阶零张量，也可以取随机数

x_tes = torch.tensor(data[3000:-1]).float().view(1,999,1)
y_tes = torch.tensor(data[3001:]).float().view(1,999,1)

output,hidden_prev = model(x_tes,hidden_prev)
hidden_prev = hidden_prev.detach()
loss = criterion(output,y_tes)
print(loss.item())

plt.plot(time_steps[3001:], y_tes.flatten(), 'r-')
plt.plot(time_steps[3001:], output.data.numpy().flatten(), 'b-')
plt.draw(); plt.pause(0.05)
