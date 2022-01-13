# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:43:36 2022

@author: Hugo
"""

import numpy as np
import pandas as pd
import os
import time 
import talib


path = '../datasets/s&p500/' #s&p500 dataset

if not os.path.exists(path+'features/'):
    os.makedirs(path+'features/')

start_time = time.time()
def file_name(file_dir):
    with open('stock_symbols.csv','w',encoding='utf-8') as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace('.csv','')+'\n' for file in files)
    return files


def stochastic_oscillator_d(df, n):
    SOK = [0]
    for i in range(n, len(df)):
        high = df.loc[(i-n):i, 'High']
        low = df.loc[(i-n):i, 'Low']
        SOK.append((df.loc[i, 'Close'] - min(low)) / (max(high) - min(low)))
    SOK = pd.Series(SOK, name='SOK')
    df = df.join(SOK)
    return df

trade_data = pd.read_csv(path+'date.csv')
trade_data.set_index('Date',inplace=True)
files = file_name(path+'pre_process/')

for file in files:
    print(file)
    pre_stock_raw = pd.read_csv(path+'pre_process/'+file)
    pre_stock = pre_stock_raw.loc[:,['Date','Open','High','Low','Close','Volume']]
    pre_stock = pre_stock.ewm(alpha=0.9).mean()
    pre_stock['MACD'], _, _ = talib.MACD(pre_stock['Close'], fastperiod=12, \
              slowperiod=26, signalperiod=9)
    pre_stock['RSI'] = talib.RSI(pre_stock['Close'], timeperiod=14)
    pre_stock['SOK'], _ = talib.STOCH(pre_stock['High'], pre_stock['Low'],\
                          pre_stock['Close'])
    pre_stock['WILLR'] =  talib.WILLR(pre_stock['High'], pre_stock['Low'],\
                          pre_stock['Close'], timeperiod=14)
    pre_stock['OBV'] = talib.OBV(pre_stock['Close'], pre_stock['Volume'])
    pre_stock['ROC'] = talib.ROC(pre_stock['Close'], timeperiod=14)

    pre_stock.loc[:,list(pre_stock.columns)] = \
        (pre_stock.loc[:,list(pre_stock.columns)] - pre_stock.loc[:,list\
                       (pre_stock.columns)].mean())/pre_stock.loc[:,\
                        list(pre_stock.columns)].std()
    pre_stock['label_1'] = pre_stock_raw['label_1']
    pre_stock['label_3'] = pre_stock_raw['label_3']
    pre_stock['label_5'] = pre_stock_raw['label_5']
    pre_stock['label_7'] = pre_stock_raw['label_7']
    pre_stock['label_9'] = pre_stock_raw['label_9']
    pre_stock['y_1'] = pre_stock_raw['y_1']
    pre_stock['y_3'] = pre_stock_raw['y_3']
    pre_stock['y_5'] = pre_stock_raw['y_5']
    pre_stock['y_7'] = pre_stock_raw['y_7']
    pre_stock['y_9'] = pre_stock_raw['y_9']
    pre_stock['label_auxiliary'] = pre_stock_raw['label_auxiliary']
    pre_stock['r_auxiliary'] = pre_stock_raw['r_auxiliary']
    pre_stock['Date'] = pre_stock_raw['Date']

    pre_stock = pre_stock.dropna()
    pre_stock.to_csv(path+'features/'+file,index=False)

print("用时：",time.time()-start_time)