# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:55:24 2021

@author: Administrator
"""
import argparse
import torch
from torch.utils.data import DataLoader
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.dataset import My_datasets
from units.log import Log
from units.initialize import initialize
from units.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM
from torchsummary import summary

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=1, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")#数据加载器的cpu线程数
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--train_txt', 
                        default="D:\\TASK\\YB\\wjr\\data\\train.txt", 
                        metavar='TT',help='the path of import train data')
    parser.add_argument('--valid_txt', 
                        default="D:\\TASK\\YB\\wjr\\data\\valid.txt", 
                        metavar='TTT',help='the path of import test data')
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cpu")
    #family
#    train_loader, val_loader = data_processing(args.mean, args.std, args.transform, train_txt_path, args.train_batch_size, args.workers, 
#                                                val_txt_path, args.test_batch_size, args.width, args.height)
    #garbage
    train_txt_path = args.train_txt
    valid_txt_path = args.valid_txt
    train_data = My_datasets(train_txt_path, True)#
    valid_data = My_datasets(valid_txt_path, False)#
    train_loader = DataLoader(dataset=train_data, pin_memory=True, batch_size=args.batch_size, shuffle=True)#
    valid_loader = DataLoader(dataset=valid_data, pin_memory=True, batch_size=args.batch_size)
    
    log = Log(log_each=3)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=3).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_data))

       # for i, (inputs, targets) in enumerate(train_loader):#
       #     inputs = inputs.cuda()
       #     targets = targets.cuda()
        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)
            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(valid_data))

        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()
    summary(model,input_size=(3,224,224))
