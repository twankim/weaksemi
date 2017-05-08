# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 20:19:24
# @Last Modified by:   twankim
# @Last Modified time: 2017-05-08 15:39:50

import numpy as np

class weakSSAC:
	def __init__(self,X,y_true,k,q=1,wtype="random"):
		self.X = X
		self.n, self.m = np.shape(X)
		self.y_true = y_true
		self.y = [0]*self.n
		self.k = k
		self.q = q
		self.eta = np.log2(self.n)
		self.beta = 1
		self.wtype = wtype
		self.wtype_list = ["random"]
		# self.wtype_list = ["random","local-distance","global-distance"]

	def set_params(self,eta,beta):
		self.eta = eta
		self.beta = beta

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
		self.clusters=[] # Assigned clusters

		for i in xrange(self.k):
			# --------------- Phase 1 ---------------
			# 1) Sample points for cluster center estimation
			if r >= len(S):
				r = len(S)
				print "Warning: sample size > remaining points"
			idx_Z = S[np.random.randint(0,len(S),r)]

			# 2) Cluster Assignment Query
			y_Z = self.clusterAssign(idx_Z)

			# Find a cluster with maximum number of samples
			p = self.clusters[np.argmax([y_Z.count(t) for t in self.clusters])]
			idx_p = idx_Z[np.array(y_Z)==p]
			mpp = np.mean(self.X[idx_p,:],axis=0)
			# print "Size of Z_p: {}".format(len(idx_p))

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
				self.y[i_assign] = p

			# 4) Exclude assigned points
			S = np.array(list(set(S)-set(idx_S_sorted[:idx_radius])))

    # Weak Same Cluster Query
	def weakQuery(self,idx_x,idx_y):
		# 1: same cluster
		# 0: not-sure
		# -1: different cluster
		if self.wtype == "random":
			return np.random.binomial(1,self.q)*\
			       2*(int(self.y_true[idx_x]==self.y_true[idx_y])-0.5)
		else:
			print "Not sure!!!"
			return 0

    # Weak Cluster Assignment Query
	def clusterAssign(self,idx_Z):
		y_Z = [0]*len(idx_Z)
		for i,idx in enumerate(idx_Z):
			if len(self.clusters)==0:
				# Currently, none of points are assigned
				answers = []
				# Assign initial point 
				y_Z[i] = 1
				self.clusters.append(1)
			else:
				# Find anchor points from each cluster (Use assignment-known points)
			    # -> Use for cluster assignment queries
				set_idx = [self.y.index(k) for k in self.clusters]
				# Ask same-cluster queries
				answers = [self.weakQuery(idx,idx_set) for idx_set in set_idx]
				if 1 in answers:
					y_Z[i] = self.y[set_idx[answers.index(1)]]
					if y_Z[i] == 0:
						print "Something Wrong!!!!!!!!! (Assigned index has cluster 0)"
				elif sum(answers) == -len(self.clusters):
					if len(self.clusters) < self.k:
						self.clusters.append(len(self.clusters)+1)
						y_Z[i] = len(self.clusters)
					# If all k same-cluster queries are not-sure
					# -> Assign as cluster 0 (Fail in cluster assignment query)
			# Update cluster assignment
			self.y[idx] = y_Z[i]
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
			else: # Not sure
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
					print "Not-sure in binary search (Fail) -> Regard it as different cluster"
					idx_r = idx_j-1
			bs_num += 1

		self.bs_num = bs_num
		return idx_l
