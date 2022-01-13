# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:41:45 2019

@author: Huho
"""

import os
import pandas as pd
import numpy as np
import time 

path = '../datasets/s&p500/' #s&p500 dataset

def file_name(file_dir):
    with open(path+'stock_symbols.csv','w',encoding='utf-8') as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace('.csv','')+'\n' for file in files)
    return files


trade_data = pd.read_csv(path+'date.csv')
trade_data.set_index('Date',inplace=True)
files = file_name(path+'pre_process')
date_close = pd.DataFrame()
date_label_1 = pd.DataFrame()
date_label_3 = pd.DataFrame()
date_label_5 = pd.DataFrame()
date_label_7 = pd.DataFrame()
date_label_9 = pd.DataFrame()
for file in files:
#    print(file)
    pre_stock = pd.read_csv(path+'pre_process/'+file)
    date_close[file.replace('.csv','')] = np.log(pre_stock['Close']/pre_stock['Close'].shift(1))
    date_label_1[file.replace('.csv','')] = pre_stock['label_1']
    date_label_3[file.replace('.csv','')] = pre_stock['label_3']
    date_label_5[file.replace('.csv','')] = pre_stock['label_5']
    date_label_7[file.replace('.csv','')] = pre_stock['label_7']
    date_label_9[file.replace('.csv','')] = pre_stock['label_9']

#Pearson CC
Cor_14 = []
for i in range(15, len(date_close)):
    Cor_14.append(date_close.loc[i-14:i].corr().values)
Cor_14 = np.array(Cor_14)
np.save(path+'graph/Cor_14.npy',Cor_14[18:])
del Cor_14
    


#Manhattan distance 
st = time.time()
MhDt_15 = []
for i in range(15, len(date_label_1)):
    mhdt_1 = np.zeros((475,475))
    for j in range(474):
        A = np.expand_dims(date_label_1.iloc[i-15:i,j].values,axis=1)\
            -date_label_1.iloc[i-15:i].values
        mhdt_1[j,:] = np.linalg.norm(A, ord=1, axis=0)
    MhDt_15.append(mhdt_1)
print(time.time()-st)
MhDt_15 = np.array(MhDt_15)
#np.save('./graph/MhDt_15.npy',MhDt_15)
MhDt_15_1=MhDt_15[18:]
np.save(path+'graph/MhDt_15.npy',MhDt_15_1)

