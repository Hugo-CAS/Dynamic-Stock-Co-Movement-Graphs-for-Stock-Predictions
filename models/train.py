# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:46:10 2022

@author: Hugo
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from utils import load_data, metrics
from HADGNN import HAD_GNN


parser = argparse.ArgumentParser('Co-movement Stock Graph for Stock tauedictions')
parser.add_argument('--path', type=str, help='path of dataset', default='../datasets/s&p500/')
parser.add_argument('--tou', type=int, help='tau-day prediction', default=9)
parser.add_argument('--input_dim', type=int, help='input dimenssions of the model', default=12)
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--seed', type=int, default=2020, help='random seed')

# hyperparameters tuning on valiadation
parser.add_argument('--lag', type=int, help='T-lag features', default=10)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--bs', type=int, help='batch size', default=5)
parser.add_argument('--dim', type=int, default=64, help='dimenssions of the model')
parser.add_argument('--out_channels', type=int, default=32, help='output dimenssions of the model')
parser.add_argument('--heads', type=int, default=1, help='number of heads')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--weight', type=float, default=5e-4, help='weight of L2 regularization')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

path_MD_15 = args.path + 'graph_date/MhDt_15/'
date_list = list(pd.read_csv(args.path + "date.csv")["Date"].iloc[42:])
tau = args.tau
lag = args.lag
train_graph_list = []

print("loading training data")
for i, date in enumerate(date_list[10:990]): #date_list[30]
    if i%100 == 0:
        print(i)
    edges, features, labels, idx_train, idx_val, idx_test = load_data(path_MD_15, 
                                                                date, lag, tau)
    graph_t = Data(x=features, edge_index=edges.t().contiguous(), y=labels)
    graph_t.train_idx  = idx_train
    graph_t.val_idx  = idx_val
    graph_t.test_idx  = idx_test
    train_graph_list.append(graph_t)

# hyperparameters tuning on valiadation
val_graph_list = []
for graph in train_graph_list[-198:]:
    val_graph_list.append(val_graph_list)


print("loading testing data")
test_graph_list = []
for i, date in enumerate(date_list[990: len(date_list)-tau]):
    if i%100 == 0:
        print(i)
    edges, features, labels, idx_train, idx_val, idx_test = load_data(path_MD_15, 
                                                                date, lag, tau)
    graph_t = Data(x=features, edge_index=edges.t().contiguous(), y=labels)
    graph_t.train_idx  = idx_train
    graph_t.val_idx  = idx_val
    graph_t.test_idx  = idx_test
    test_graph_list.append(graph_t)




random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

model = HAD_GNN(in_dim=args.input_dim, hid_dim=args.dim, in_channels=args.dim, out_channels=args.out_channels, heads=args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
train_loader = DataLoader(train_graph_list, batch_size=args.bs, shuffle=False)     
test_loader = DataLoader(test_graph_list, batch_size=args.bs)     

num_break = 0
max_acc = 0
max_mcf = 0
early_stopping = 10

train_loss = []
time_recorder = []
   
acc_list = []
macro_f1_list = [] 

for epoch in range(args.n_epoch):
    t0 = time.time()
    
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    t1 = time.time()
    time_recorder.append(t1-t0)
    print("time of train epoch:", t1 - t0)
    
    model.eval()
    
    train_label_list = []
    train_pred_list = []
    for tdata in train_loader:
        train_label_list.extend(tdata.y.detach().cpu().numpy())
        tdata = tdata.to(device)
        _, tr_pred = model(tdata).max(dim=1)
        train_pred_list.extend(tr_pred.detach().cpu().numpy())
    train_acc, train_mac_f1 = metrics(train_pred_list, train_label_list)
    
    label_list = []
    pred_list = []
    for tdata in test_loader:
        label_list.extend(tdata.y.detach().cpu().numpy())
        tdata = tdata.to(device)
        _, pred = model(tdata).max(dim=1)
        pred_list.extend(pred.detach().cpu().numpy())
    
    acc, mac_f1 = metrics(pred_list, label_list)
    acc_list.append(acc)
    macro_f1_list.append(mac_f1)
    
    print("time of val epoch:", time.time() - t1)
    print('Epoch {:3d},'.format(epoch+1),
          'Train Accuracy {:.4f}'.format(train_acc),
          'Train Macro_f1 {:.4f}'.format(train_mac_f1),
          'Accuracy {:.4f},'.format(acc),
          'Macro_f1 {:.4f},'.format(mac_f1),
          'time {:4f}'.format(time.time()-t0))
    
    #early stopping
    #hyperparameters tuning on valiadation
    if train_acc < max_acc and train_mac_f1 < max_mcf:
        num_break += 1
        if num_break >= early_stopping:
            break
    else:
        num_break = 0
        if train_acc > max_acc:
            max_acc = train_acc
            if not os.path.exists('./save_model'):
                os.makedirs('./save_model')
            torch.save(model.state_dict(), './save_model/{}.pkl'.format(str(args.lr)+'_'+str(args.bs)+'_'+str(epoch)))
        if train_mac_f1 > max_mcf:
            max_mcf = train_mac_f1              