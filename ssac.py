# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 20:19:24
# @Last Modified by:   twankim
# @Last Modified time: 2017-10-18 23:42:43

import numpy as np

class weakSSAC:
    def __init__(self,X,y_true,k,q=1,wtype="random",ris=None):
        self.X = X
        self.n, self.m = np.shape(X)
        self.y_true = y_true
        self.y = [0]*self.n
        self.k = k
        self.q = 1
        self.eta = np.log2(self.n)
        self.beta = 1
        self.wtype = wtype
        self.wtype_list = ["random","local","global"]
        self.ris = ris
        self.centers = [np.mean(X[self.y_true==i],axis=0) for i in xrange(1,k+1)]

    def set_params(self,eta,beta,q=1,rho=0.8,nu=1.25):
        self.eta = eta
        self.beta = beta
        self.q = q
        self.rho = rho
        self.nu = nu

    def set_wtype(self,wtype):
        assert wtype in self.wtype_list,\
                "Weakness type {} is not supported. Supported {}".format(
                                 wtype,self.wtype_list)
        self.wtype = wtype

    def fit(self):
        self.y = [0]*self.n # Initialize cluster assignments as 0
        S = np.arange(self.n) # Initial set of indices
        r = int(np.ceil(self.k*self.eta)) # Sample size for phase 1
        self.mpps = [] # Estimated cluster centers
        self.labels = [] # Cluster labels which are recovered (correspond to mpps)
        self.clusters = [] # Cluster labels which are known

        for i_k in xrange(self.k):
            is_overwrite = False # Flag for overwriting assignment
            # --------------- Phase 1 ---------------
            # 1) Sample points for cluster center estimation
            if r >= len(S):
                r = len(S)
                print "    Warning: sample size > remaining points"
            idx_Z = S[np.random.randint(0,len(S),r)]

            # 2) Cluster Assignment Query
            y_Z = self.clusterAssign(idx_Z)
            if sum(y_Z)==0:
                if self.wtype == 'random':
                    print "    !!! Not-sure in Cluster Assignment. (q,eta,beta)=({},{},{}), i_k={}"\
                           .format(self.q,self.eta,self.beta,i_k)
                elif self.wtype == 'local':
                    print "    !!! Not-sure in Cluster Assignment. (nu,rho,eta,beta)=({},{},{},{}), i_k={}"\
                        .format(self.nu,self.rho,self.eta,self.beta,i_k)
                elif self.wtype == 'global':
                    print "    !!! Not-sure in Cluster Assignment. (rho,eta,beta)=({},{},{}), i_k={}"\
                        .format(self.rho,self.eta,self.beta,i_k)

            else:
                # Find a cluster with maximum number of samples
                p = self.clusters[np.argmax([y_Z.count(t) for t in self.clusters])]
                idx_p = idx_Z[np.array(y_Z)==p]
                mpp = np.mean(self.X[idx_p,:],axis=0)
                # print "Size of Z_p: {}".format(len(idx_p))
                if p in self.labels:
                    if self.wtype == 'random':
                        print "    !!! Cluster p revisited (q,eta,beta)=({},{},{}), i_k={}".format(
                                  self.q,self.eta,self.beta,i_k)
                    elif self.wtype == 'local':
                        print "    !!! Cluster p revisited (nu,rho,eta,beta)=({},{},{},{}), i_k={}".format(
                                  self.nu,self.rho,self.eta,self.beta,i_k)
                    elif self.wtype == 'global':
                        print "    !!! Cluster p revisited (rho,eta,beta)=({},{},{}), i_k={}".format(
                                  self.rho,self.eta,self.beta,i_k)

                    self.mpps[self.labels.index(p)] = mpp # Update mean
                else:
                    self.mpps.append(mpp) # Estimated cluster center
    
                # --------------- Phase 2 ---------------
                # 1) Sort points in S based on distances from the cluster center.
                idx_S_sorted = S[np.argsort(np.linalg.norm(
                                              self.X[S,:]-np.tile(mpp,(len(S),1)),
                                              axis=1))]
                # 2) Apply binary serach algorithm
                idx_radius = self.binarySearch(idx_S_sorted,idx_p)
    
                # 3) Assign clusters based on the radius.
                for i_assign in idx_S_sorted[:idx_radius]:
                    if self.y[i_assign] not in [0,p]:
                        is_overwrite = True
                    self.y[i_assign] = p
                
                if not p in self.labels:
                    self.labels.append(p)

                # Handle overwrited cluster assignments
                # Binary Search can overwrite some pre-assigned points in Cluster-Assignment
                if is_overwrite:
                    deleted_clusters = set(self.clusters)-set(self.y)
                    if len(deleted_clusters)>0:
                        print "    Assignment Erased!!!!!!!"
                        # Update self.clusters to delete erased clusters
                        self.clusters = [cluster for cluster in self.clusters \
                                            if (cluster not in deleted_clusters)]

                        # Update self.labels and self.mpps if affected
                        self.mpps = [self.mpps[j] for j in xrange(len(self.mpps)) \
                                        if (self.labels[j] not in deleted_clusters)]
                        self.labels = [label for label in self.labels \
                                        if (label not in deleted_clusters)]
                
                # 4) Exclude assigned points
                S = np.array(list(set(S)-set(idx_S_sorted[:idx_radius])))

        # Some Failure cases
        if len(self.labels) < self.k: # Number of recovered clusters are less than k
            if self.wtype == 'random':
                print "    !!! Number of assigned clusters={} is less than k={}. (q,eta,beta)=({},{},{})"\
                      .format(len(self.labels),self.k,self.q,self.eta,self.beta)
            elif self.wtype == 'local':
                print "    !!! Number of assigned clusters={} is less than k={}. (nu,rho,eta,beta)=({},{},{})"\
                      .format(len(self.labels),self.k,self.nu,self.rho,self.eta,self.beta)
            elif self.wtype == 'global':
                print "    !!! Number of assigned clusters={} is less than k={}. (rho,eta,beta)=({},{})"\
                      .format(len(self.labels),self.k,self.rho,self.eta,self.beta)
            return False
        else:
            return True

    # Weak Same Cluster Query
    def weakQuery(self,idx_x,idx_y):
        # 1: same cluster
        # 0: not-sure
        # -1: different cluster
        if self.wtype == "random":
            return np.random.binomial(1,self.q)*\
                   2*(int(self.y_true[idx_x]==self.y_true[idx_y])-0.5)
        elif self.wtype == "local":
            d_xy = np.linalg.norm(self.X[idx_x,:]-self.X[idx_y,:])
            if self.y_true[idx_x] == self.y_true[idx_y]:
                if d_xy > 2*self.rho*self.ris[self.y_true[idx_x]-1]:
                    return 0
            else:
                d_xi = np.linalg.norm(self.X[idx_x,:]-self.centers[self.y_true[idx_x]-1])
                d_yi = np.linalg.norm(self.X[idx_y,:]-self.centers[self.y_true[idx_y]-1])
                if (self.nu-1)*min(d_xi,d_yi)>d_xy:
                    return 0
            return 2*(int(self.y_true[idx_x]==self.y_true[idx_y])-0.5)
        elif self.wtype == "global":
            d_xy = np.linalg.norm(self.X[idx_x,:]-self.X[idx_y,:])
            if self.y_true[idx_x] == self.y_true[idx_y]:
                if d_xy > 2*self.rho*self.ris[self.y_true[idx_x]-1]:
                    return 0
            else:
                d_xi = np.linalg.norm(self.X[idx_x,:]-self.centers[self.y_true[idx_x]-1])
                d_yi = np.linalg.norm(self.X[idx_y,:]-self.centers[self.y_true[idx_y]-1])
                if (d_xi > self.rho*self.ris[self.y_true[idx_x]-1]) | \
                   (d_yi > self.rho*self.ris[self.y_true[idx_y]-1]):
                    return 0
            return 2*(int(self.y_true[idx_x]==self.y_true[idx_y])-0.5)
        else:
            print "Not sure!!!"
            return 0

    # Weak Cluster Assignment Query
    def clusterAssign(self,idx_Z):
        y_Z = [0]*len(idx_Z)
        for i,idx in enumerate(idx_Z):
            if len(self.clusters)==0: # Currently, none of points are assigned
                answers = []
                # Assign initial point
                y_Z[i] = 1
                self.clusters.append(1)
                self.y[idx] = y_Z[i] # Update cluster assignment
            else:
                # Find anchor points from each cluster (Use assignment-known points)
                # -> Use for cluster assignment queries
                set_idx = [self.y.index(k) for k in self.clusters]
                # Ask same-cluster queries
                answers = [self.weakQuery(idx,idx_set) for idx_set in set_idx]
                if 1 in answers:
                    y_Z[i] = self.y[set_idx[answers.index(1)]]
                elif sum(answers) == -len(self.clusters):
                    if len(self.clusters) < self.k:
                        y_Z[i] = [k_n for k_n in xrange(1,self.k+1) if k_n not in self.clusters][0]
                        self.clusters.append(y_Z[i])
                        self.y[idx] = y_Z[i] # Update cluster assignment
                    else:
                        print "    !!!! Something Wrong. Got k Q=-1 Answers."
                # If k same-cluster queries ended up as not-sure
                # -> Assign as cluster 0 (Fail in cluster assignment query)
        return y_Z

    # Binary Search Algorithm for SSAC
    def binarySearch(self,idx_S_sorted,idx_p):
        idx_l = 0 # left index
        idx_r = len(idx_S_sorted)-1 # right index

        bs_num = 0 # Number of binary search comparison
        while idx_l <= idx_r:
            idx_j = int(np.floor((idx_l+idx_r)*0.5))
            
            # Use the closest point as an anchor point for same-cluster query
            answer = self.weakQuery(idx_S_sorted[0],idx_S_sorted[idx_j])
            if answer == 1: # Same cluster
                idx_l = idx_j+1
            elif answer == -1: # Different cluster
                idx_r = idx_j-1
            else:
                # Not sure
                # Sample beta-1 additional points
                set_idx_B = np.random.randint(0,len(idx_p),self.beta-1)
                answers = [self.weakQuery(idx_p[idx_B],
                                          idx_S_sorted[idx_j]) for idx_B in set_idx_B]
                if 1 in answers:
                    idx_l = idx_j+1
                elif -1 in answers:
                    idx_r = idx_j-1
                else:
                    # Can return fail, but regard it as not in the same cluster
                    # Follows the unified-weak BinarySearch model.
                    print "    Not-sure in Binary Search. -> Regard as different clusters"
                    idx_r = idx_j-1
            bs_num += 1

        self.bs_num = bs_num
        return idx_l
