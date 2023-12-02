#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:06:19 2017

@author: quien
"""

import glob

import numpy as np
import numpy.random as rd

import matplotlib.image as img
import matplotlib.pyplot as plt

names = glob.glob("train_set/face/*.pgm")

X = None
d = None
for name in names:
    a = img.imread(name)/255.0
    d = a.shape
    a = a.reshape((a.shape[0]*a.shape[1],1))
    if X is None:
        X = a
    else:
        X = np.concatenate((X,a),axis=1)

k_N = 10
k_M = 10
K = k_N*k_M

Y = np.zeros(X.shape)
W = rd.rand(X.shape[0],K)
H = rd.rand(K,X.shape[1])
H = H / np.repeat(np.sum(H,axis=1).reshape((H.shape[0],1)),H.shape[1],axis=1)

T = 1000
E = np.zeros(T)

for t in range(T):
    Y = np.dot(W,H)
    E[t] = np.sum(X*np.log(X/Y+1e-10)-X+Y)
    print(t)
    Z = X/Y
    Wones = np.repeat(np.sum(W,axis=0).reshape((K,1)),H.shape[1],axis=1)
    onesH = np.repeat(np.sum(H,axis=1).reshape((1,K)),W.shape[0],axis=0)
    
    H_n = H * np.dot(W.T,Z)/Wones 
    W_n = W * np.dot(Z,H.T)/onesH
    
    H = H_n / np.repeat(np.sum(H_n,axis=1).reshape((H.shape[0],1)),H.shape[1],axis=1)
    W = np.copy(W_n)
    
plt.plot(E)
f,axarr = plt.subplots(k_N,k_M)
for i in range(k_N):
    for j in range(k_M):
        axarr[i,j].imshow(W[:,k_N*i+j].reshape(d),cmap='gnuplot')
        axarr[i,j].set_xticklabels([])
        axarr[i,j].set_yticklabels([])
        axarr[i,j].grid(False)
plt.show()