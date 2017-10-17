# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 00:06:53
# @Last Modified by:   twankim
# @Last Modified time: 2017-10-16 23:52:22

import numpy as np
from sklearn.datasets import make_blobs

class genData:
    def __init__(self,n,m,k,min_gamma=1,max_gamma=1.25,std=1.0):
        self.n = n
        self.m = m
        self.k = k
        self.std = std
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.gamma = 0
        self.ris = np.zeros(self.k)

    def gen(self):
        while (self.gamma <= self.min_gamma) or (self.gamma > self.max_gamma):
            X,y = make_blobs(n_samples=self.n,n_features=self.m,centers=self.k,cluster_std = self.std)
            X_means = [X[y==t,:].mean(axis=0) for t in xrange(self.k)]
            gammas = []

            for i in xrange(self.k):
                ri = max(np.linalg.norm(X[y==i,:]-np.tile(X_means[i],(sum(y==i),1)),axis=1))
                ra = min(np.linalg.norm(X[y!=i,:]-np.tile(X_means[i],(sum(y!=i),1)),axis=1))
                gammas.append(ra/float(ri))
                self.ris[i] = ri

            y += 1
            self.gamma = min(gammas)

        self.X = X
        self.y = y
        return X,y

    def calc_eta(self,q,delta):
        assert (q >0) and (q<=1), "q must be in (0,1]"
        if q < 1:
            return int(np.ceil(np.log(2.0*self.k*(self.m+1)/delta) / \
                               np.log(1.0/(1 - q**(self.k-1)*(1-np.exp(-(self.gamma-1)**2 /8.0))))
                               ))
        else:
            return int(np.ceil( 8*np.log(2*self.k*(self.m+1)/delta) / (self.gamma-1)**2 ))
    
    def calc_beta(self,q,delta):
        assert (q >0) and (q<=1), "q must be in (0,1]"
        if q < 1:
            return int(np.ceil(np.log(2*self.k*np.log(self.n)/delta) / np.log(1.0/(1-q))))
        else:
            return 1
