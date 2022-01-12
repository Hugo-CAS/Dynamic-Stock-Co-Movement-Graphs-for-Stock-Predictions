# HAD-GNN
This repository contains the data for the paper "Inductive Representation Learning on Dynamic Stock Co-Movement Graphs for Stock Predictions". The code will be available later.

# Datasets
This paper collected datasets from Yahoo! Finance for three major stock markets in the US, China, 
and Australia. The datasets "s&p500", "csi300", and "asx300" represent major companies in these market indices: Standard and
Poor’s 500 (S&P 500), China Securities Index 300 (CSI 300), and Australian Securities 
Exchange 300 (ASX 300). In each dataset folder, there are two types of files for each dataset.

The file "./raw_data/stock_name.csv" contains the raw data for stock "stock_name" on the given period. The raw data includes five features: the opening price, high price, low price, closing price, and trading volume.
The file "date.csv" gives the data collection period.
The file "stock_symbols" is the stock list of the corresponding market index.