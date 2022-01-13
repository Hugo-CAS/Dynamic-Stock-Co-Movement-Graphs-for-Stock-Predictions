# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:30:25 2022

@author: Hugo
"""

import os
import pandas as pd
import numpy as np

path = '../datasets/s&p500/' #s&p500 dataset
if not os.path.exists(path+'pre_process/'):
    os.makedirs(path+'pre_process/')
    
def file_name():
    with open(path+'stock_symbols.csv','r',encoding='utf-8') as fr:
        stock_symbols = fr.readlines()
        stock_symbols = [sy.strip()+'.csv' for sy in stock_symbols]
    return stock_symbols


trade_data = pd.read_csv(path+'date.csv')
trade_data.set_index('Date',inplace=True)
files = file_name()
files_filter = files.copy()
L = []
Clp_ratio_1 = []
Clp_ratio_3 = []
Clp_ratio_5 = []
Clp_ratio_7 = []
Clp_ratio_9 = []
Clp_ratio_a = []
for i, file in enumerate(files):
    if i%10 == 0:
        print(i)
    raw_stock = pd.read_csv(path+'raw_data/'+file) 
    stock_df = raw_stock[['Date','Open','High','Low','Close','Volume']]
    stock_df.columns = ['Date','Open','High','Low','Close','Volume']

    stock_df = stock_df.set_index('Date')
    res = stock_df.copy()
    row = np.where(np.isnan(res))
    for r in row[0]:
        res.loc[res.index[r]] = res.loc[res.index[r-1]] 
    L.append(len(res.index))
    clp_ratio_1 = []
    clp_ratio_3 = []
    clp_ratio_5 = []
    clp_ratio_7 = []
    clp_ratio_9 = []

    clp_ratio_1 = res['Close'].shift(-1)/res['Close']-1
    clp_ratio_3 = res['Close'].shift(-3)/res['Close']-1
    clp_ratio_5 = res['Close'].shift(-5)/res['Close']-1
    clp_ratio_7 = res['Close'].shift(-7)/res['Close']-1
    clp_ratio_9 = res['Close'].shift(-9)/res['Close']-1
    clp_ratio_a = res['Close']/res['Close'].shift(5)-1
    label_1 = np.zeros(len(clp_ratio_1))
    label_3 = np.zeros(len(clp_ratio_1))
    label_5 = np.zeros(len(clp_ratio_1))
    label_7 = np.zeros(len(clp_ratio_1))
    label_9 = np.zeros(len(clp_ratio_1))
    label_auxiliary = np.zeros(len(clp_ratio_1))
    Clp_ratio_1 = Clp_ratio_1 + list(clp_ratio_1)[5:-10]
    Clp_ratio_3 = Clp_ratio_3 + list(clp_ratio_3)[5:-10]
    Clp_ratio_5 = Clp_ratio_5 + list(clp_ratio_5)[5:-10]
    Clp_ratio_7 = Clp_ratio_7 + list(clp_ratio_7)[5:-10]
    Clp_ratio_9 = Clp_ratio_9 + list(clp_ratio_9)[5:-10]
    Clp_ratio_a = Clp_ratio_a + list(clp_ratio_a)[5:-10]
    for i in range(len(clp_ratio_1)):
        if clp_ratio_1[i] is np.nan:
            label_1[i] = 0
        else:
            
            #set different thresholds for different datasets
            
            if clp_ratio_1[i] <= -0.00499:
                label_1[i] = -1
            elif clp_ratio_1[i] > 0.005:
                label_1[i] = 1
            else:
                label_1[i] = 0
                
        if clp_ratio_3[i] is np.nan:
            label_3[i] = 0
        else:
            if clp_ratio_3[i] <= -0.00699:
                label_3[i] = -1
            elif clp_ratio_3[i] > 0.010:
                label_3[i] = 1
            else:
                label_3[i] = 0
                
        if clp_ratio_5[i] is np.nan:
            label_5[i] = 0
        else:
            if clp_ratio_5[i] <= -0.008499:
                label_5[i] = -1
            elif clp_ratio_5[i] > 0.0135:
                label_5[i] = 1
            else:
                label_5[i] = 0  
                
        if clp_ratio_7[i] is np.nan:
            label_7[i] = 0
        else:
            if clp_ratio_7[i] <= -0.01049: 
                label_7[i] = -1
            elif clp_ratio_7[i] > 0.017:
                label_7[i] = 1
            else:
                label_7[i] = 0  
                
        if clp_ratio_9[i] is np.nan:
            label_9[i] = 0
        else:
            if clp_ratio_9[i] <= -0.01149:
                label_9[i] = -1
            elif clp_ratio_9[i] > 0.0200:
                label_9[i] = 1
            else:
                label_9[i] = 0 
                
        if clp_ratio_a[i] is np.nan:
            label_auxiliary[i] = 0
        else:
            if clp_ratio_a[i] <= -0.008499:
                label_auxiliary[i] = -1
            elif clp_ratio_a[i] > 0.014:
                label_auxiliary[i] = 1
            else:
                label_auxiliary[i] = 0 
    res['label_1'] = label_1
    res['label_3'] = label_3
    res['label_5'] = label_5
    res['label_7'] = label_7
    res['label_9'] = label_9
    res['label_auxiliary'] = label_auxiliary
    res['y_1'] = clp_ratio_1
    res['y_3'] = clp_ratio_3
    res['y_5'] = clp_ratio_5
    res['y_7'] = clp_ratio_7
    res['y_9'] = clp_ratio_9
    res['r_auxiliary'] = clp_ratio_a
    res.to_csv(path+'pre_process/'+file)

