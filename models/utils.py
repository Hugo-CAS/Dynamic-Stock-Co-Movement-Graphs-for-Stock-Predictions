# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:54:57 2022

@author: Hugo
"""

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time 

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path, date, lf=10, pr=1):
    """Load stock network dataset"""
    labels_pre = np.genfromtxt(path+'labels/'+date+"_graph_label_"+str(pr)+".txt")
    
    features = np.load(path+'features_'+str(lf)+'/'+date+'_features_'+str(lf)+'.npy')
    
    edges_0 = np.genfromtxt(path+'edges/'+date+"_graph.txt", dtype = np.int32)
    edges_1 = np.concatenate(([edges_0[:,1]], [edges_0[:,0]]), axis=0).T
   
    edges = np.concatenate((edges_0, edges_1), axis=0)
    edges = torch.LongTensor(edges)
    
    idx_train = torch.tensor(np.arange(edges.shape[0]), dtype=torch.long)
    idx_val = idx_train#.copy()
    idx_test = idx_train#.copy()

    features = torch.FloatTensor(np.array(features))
    labels_pre = torch.LongTensor(labels_pre + 1)
    return edges, features, labels_pre, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def metrics(preds, labels):
    labels = np.array(labels)
    preds = np.array(preds)
    acc = accuracy_score(labels,preds)
    macro_f1=f1_score(labels,preds,average='macro')
    return acc, macro_f1
    