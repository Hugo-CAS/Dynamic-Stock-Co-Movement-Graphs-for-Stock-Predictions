# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:54:44 2022

@author: Hugo
"""

import numpy as np
import pandas as pd
import os
import time

path = '../datasets/s&p500/' #s&p500 dataset

if not os.path.exists(path+'graph_date/PsDt_15/edges'):
    os.makedirs(path+'graph_date/PsDt_15/edges')
    
if not os.path.exists(path+'graph_date/PsDt_15/Adjs'):
    os.makedirs(path+'graph_date/PsDt_15/Adjs')
    
if not os.path.exists(path+'graph_date/PsDt_15/features_10'):
    os.makedirs(path+'graph_date/PsDt_15/features_10')
    
if not os.path.exists(path+'graph_date/PsDt_15/labels'):
    os.makedirs(path+'graph_date/PsDt_15/labels')
    
date = pd.read_csv(path+'date.csv')
PsDt_15 = np.load(path+'graph/PsDt_15.npy')
edges = []

def file_name():
    with open(path+'stock_symbols.csv','r',encoding='utf-8') as fr:
        stock_symbols = fr.readlines()
        stock_symbols = [sy.strip()+'.csv' for sy in stock_symbols]
    return stock_symbols
files = file_name()

symbol_features = {}

s_t = time.time()
for file in files:
    features_df = pd.read_csv(path+'features/'+file)
    symbol_features[file] = features_df

date_15 = pd.read_csv(path+'features/AAPL.csv')['Date']

Adj = []
tou = 9 #1,3,5,7,9
P_num_edge = []
N_num_edge = []
for i in range(tou, PsDt_15.shape[0]-tou):
    if i%50 == 0:
        print(i)
    date_feature_10 = []
    date_feature_20 = []
    date_feature_40 = []
    label_aux = []
    p_num_edge = 0
    n_num_edge = 0
    
    adj = np.zeros((475,475))
    with open(path+'graph_date/PsDt_15/edges/'+str(date_15.loc[i])+'_graph.txt','w',encoding='utf-8') as fw:
        for j in range(475):
            for k in range(0, 475):
                if PsDt_15[i,j,k] > 0.80:
                    if k != j:
                        adj[j,k] = 1
                    if k >= j+1:
                        fw.write(str(j)+' '+str(k)+'\n')
                        p_num_edge += 1
                        
                if PsDt_15[i,j,k] < -0.80:
                    if k != j:
                        adj[j,k] = -1
                    if k >= j+1:
                        fw.write(str(j)+' '+str(k)+'\n')
                        n_num_edge += 1
    # P_num_edge.append(p_num_edge)
    # N_num_edge.append(n_num_edge)
    # Adj.append(adj)
    np.save(path+'graph_date/PsDt_15/Adjs/'+str(date_15.loc[i])+'_Adj.npy', adj)
    # edges.append(num_edge)
    # d = str(date_15.loc[i])
    
    with open (path+'graph_date/PsDt_15/labels/'+str(date_15.loc[i])+'_graph_label_1.txt','w',encoding='utf-8') as fw_1:
        with open (path+'graph_date/PsDt_15/labels/'+str(date_15.loc[i])+'_graph_label_3.txt','w',encoding='utf-8') as fw_3:
            with open (path+'graph_date/PsDt_15/labels/'+str(date_15.loc[i])+'_graph_label_5.txt','w',encoding='utf-8') as fw_5:
                with open (path+'graph_date/PsDt_15/labels/'+str(date_15.loc[i])+'_graph_label_7.txt','w',encoding='utf-8') as fw_7:
                    with open (path+'graph_date/PsDt_15/labels/'+str(date_15.loc[i])+'_graph_label_9.txt','w',encoding='utf-8') as fw_9:
                        with open(path+'graph_date/PsDt_15/labels/'+str(date_15.loc[i])+'_graph_label_aux.txt','w',encoding='utf-8') as fl:
                            for file in files:
                                features_df = symbol_features[file]
                                label_1 = features_df.loc[i,'y_1']
                                label_3 = features_df.loc[i,'y_3']
                                label_5 = features_df.loc[i,'y_5']
                                label_7 = features_df.loc[i,'y_7']
                                label_9 = features_df.loc[i,'y_9']
                                label_aux = features_df.loc[i,'label_auxiliary']
                                fw_1.write(str(label_1)+'\n')
                                fw_3.write(str(label_3)+'\n')
                                fw_5.write(str(label_5)+'\n')
                                fw_7.write(str(label_7)+'\n')
                                fw_9.write(str(label_9)+'\n')
                                fl.write(str(label_aux)+'\n')
                                date_feature_10.append(features_df.loc[i-9:i,['Open','High',\
                                                                    'Low','Close','Volume','MACD',\
                                                                    'RSI','SOK','WILLR','OBV','ROC',\
                                                                    'r_auxiliary']].values)
                                # if i >= 19:
                                #       date_feature_20.append(features_df.loc[i-19:i,['Open','High',\
                                #                                     'Low','Close','Volume','MACD',\
                                #                                     'RSI','SOK','WILLR','OBV','ROC',\
                                #                                     'r_auxiliary']].values)
                                # if i >= 39:
                                #     date_feature_40.append(features_df.loc[i-39:i,['Open','High',\
                                #                                     'Low','Close','Volume','MACD',\
                                #                                     'RSI','SOK','WILLR','OBV','ROC',\
                                #                                     'r_auxiliary']].values)
        
    date_feature_10 = np.array(date_feature_10)
    np.save(path+'graph_date/PsDt_15/features_10/'+str(date_15.loc[i])+'_features_10.npy', date_feature_10)
    # if i >= 19:
    #     date_feature_20 = np.array(date_feature_20)
    #     np.save('./graph_date/PsDt_15/features_20/'+str(date_15.loc[i])+'_features_20.npy', date_feature_20)
    # if i >= 39:
    #     date_feature_40 = np.array(date_feature_40)
    #     np.save('./graph_date/PsDt_15/features_40/'+str(date_15.loc[i])+'_features_40.npy', date_feature_40)

        
print('用时:', time.time()-s_t)      
       




