#%% -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:43:56 2019

@author: chahida
Email: abderrazak.chahid@gmail.com
"""
#% Packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#%% ##################################################################################################
def Set_levels_Sigma(k,M,mu,sigma):

    Levels= np.asarray([i for i in range(1,M+1)])
    N=(M-1)//2;
#    VECTOR=[-floor(N/2): floor(N/2)];
    VECTOR=[i for i in range(-N,N + 1)]
#    Level_intervals= mu+k*sigma*VECTOR;
    Level_intervals=[ mu+k*sigma*i for i in VECTOR]
    return Levels , Level_intervals

def mapping_levels(X,Level_intervals, Levels):

    Q=np.zeros(shape = (X.shape[0],X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xn=X[i,j]
            Q[i,j]=Get_level(Xn,Level_intervals,Levels);

    return Q


def Get_level(Vx,Level_intervals,Levels):
#    idx=find(Vx<=Level_intervals);
    idx=[ i for i in range(len(Level_intervals)) if Vx <=Level_intervals[i] ]
#    print('idx',idx)
    if len(idx)==0:
        L=Levels[-1:];
    else:
       l=idx[0];
       L=Levels[l];

    return np.asscalar(np.array(L))



