# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 00:06:53
# @Last Modified by:   twankim
# @Last Modified time: 2017-10-24 03:23:56

import numpy as np

class DATASET:
    def __init__(self,X,y,X_org=None,dict_label=None):
        self.X = X
        self.y = y
        self.n,self.m = np.shape(self.X)
        self.k = len(np.unique(self.y))
        self.gamma = 0
        self.ris = np.zeros(self.k)
        self.init_fun()
        self.X_org = X_org
        self.dict_label = dict_label

    def init_fun(self):
        self.X_means = [self.X[self.y==t,:].mean(axis=0) for t in xrange(self.k)]
        gammas = []

        for i in xrange(self.k):
            ri = max(np.linalg.norm(
                        self.X[self.y==i,:]-np.tile(self.X_means[i],(sum(self.y==i),1)),
                        axis=1))
            ra = min(np.linalg.norm(
                        self.X[self.y!=i,:]-np.tile(self.X_means[i],(sum(self.y!=i),1)),
                        axis=1))
            gammas.append(ra/float(ri))
            self.ris[i] = ri

        self.y += 1
        self.gamma = min(gammas)

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
