# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 00:06:53
# @Last Modified by:   twankim
# @Last Modified time: 2017-10-19 23:50:33

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
        self.X_means = X_means
        return self.X,self.y,self.ris

    def calc_eta(self,delta,weak='random',q=1.0,nu=None,rho=None):
        assert (q >0) and (q<=1), "q must be in (0,1]"
        assert weak in ['random','local','global'], \
                    "weak must be in ['random','local','global']"

        if weak == 'random':
            if q < 1:
                return int(np.ceil(np.log(2.0*self.k*(self.m+1)/delta) / \
                                   np.log(1.0/(1 - q**(self.k-1)*(1-np.exp(-(self.gamma-1)**2 /8.0))))
                                   ))
            else:
                return int(np.ceil( 8*np.log(2*self.k*(self.m+1)/delta) / (self.gamma-1)**2 ))
        elif weak == 'local':
            c_param = min(2*rho-1,self.gamma-nu+1)
            qds = []
            for i in xrange(self.k):
                dists = np.linalg.norm(
                            self.X[self.y==i+1,:]-np.tile(self.X_means[i],(sum(self.y==i+1),1)),
                            axis=1)
                qds.append(sum(dists<c_param*self.ris[i])/float(len(dists)))

            q = min(qds)
            return int(np.ceil(np.log(2.0*self.k*(self.m+1)/delta) / \
                               np.log(1.0/(1 - q**(self.k-1)*(1-np.exp(-(self.gamma-1)**2 /8.0))))
                               ))
        elif weak == 'global':
            c_param = 2*rho-1
            qds = []
            for i in xrange(self.k):
                dists = np.linalg.norm(
                            self.X[self.y==i+1,:]-np.tile(self.X_means[i],(sum(self.y==i+1),1)),
                            axis=1)
                qds.append(sum(dists<c_param*self.ris[i])/float(len(dists)))

            q = min(qds)
            return int(np.ceil(np.log(2.0*self.k*(self.m+1)/delta) / \
                               np.log(1.0/(1 - q**(self.k-1)*(1-np.exp(-(self.gamma-1)**2 /8.0))))
                               ))
        else:
            return int(np.ceil( 8*np.log(2*self.k*(self.m+1)/delta) / (self.gamma-1)**2 ))
    
    def calc_beta(self,delta,weak='random',q=1.0,nu=None,rho=None):
        assert (q >0) and (q<=1), "q must be in (0,1]"
        assert weak in ['random','local','global'], \
                    "weak must be in ['random','local','global']"
        if weak == 'random':
            if q < 1:
                return int(np.ceil(np.log(2*self.k*np.log(self.n)/delta) / np.log(1.0/(1-q))))
        return 1
