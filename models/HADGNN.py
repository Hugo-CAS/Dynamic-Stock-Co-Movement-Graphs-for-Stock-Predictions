# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:47:01 2022

@author: Hugo
"""

import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

lag = 10
class time_att(nn.Module):
    def __init__(self, n_hidden_1):
        super(time_att, self).__init__()
        self.W = nn.Parameter(torch.zeros(lag, n_hidden_1))
        nn.init.xavier_normal_(self.W.data)
    def forward(self, ht):
        ht_W = ht.mul(self.W)
        ht_W = torch.sum(ht_W, dim=2)
        att = F.softmax(ht_W, dim=1)
        return att

class ATT_LSTM(nn.Module):
    def __init__(self, in_dim=12, n_hidden_1=64, out_dim=64):
        super(ATT_LSTM, self).__init__()
        self.LSTM = nn.LSTM(in_dim, n_hidden_1, 1, batch_first=True, bidirectional = False)
        self.time_att = time_att(n_hidden_1)
        self.fc = nn.Sequential(nn.Linear(n_hidden_1, out_dim), nn.ReLU(True))
        
    def forward(self, x):
        ht, (hn, cn) = self.LSTM(x)
        t_att = self.time_att(ht).unsqueeze(dim=1)
        att_ht = torch.bmm(t_att, ht)
        att_ht = self.fc(att_ht)
        
        return att_ht
    
class HAD_GNN(torch.nn.Module):
    def __init__(self, in_dim=12, hid_dim=64, in_channels=64, out_channels=32, heads=1, num_classes=3):
        super(HAD_GNN, self).__init__()
        self.dropout = 0.25
        self.training = True
        self.att_lstm = ATT_LSTM(in_dim, hid_dim, hid_dim)
        self.gatconv_1 = GATConv(in_channels, out_channels, heads, dropout=self.dropout)
        self.gatconv_2 = GATConv(out_channels*heads+in_dim, out_channels, 1, dropout=self.dropout)
        self.linear = nn.Linear(out_channels, num_classes)
        self.act = nn.ReLU()
    
    def forward(self, data):
        x0, edge_index = data.x, data.edge_index
        x = self.att_lstm(x0)
        x = torch.squeeze(x)
        x = F.dropout(x,  self.dropout, training=self.training)
        x = self.gatconv_1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gatconv_2(torch.cat((x,x0[:,-1,:]),dim=1), edge_index)
        x = self.act(self.linear(x))
        
        return F.log_softmax(x, dim=1)
 