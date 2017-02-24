import numpy as np

# wtype_list = ["random","local","global"]
wtype_list = ["random"]

class weakSSAC:
	def __init__(self,X,k,q=1,wtype="random"):
		self.X = X
		self.m, self.n = np.shape(X)
		self.k = k
		self.q = q
		self.eta = np.log2(self.n)
		self.beta = 1
		se.f.wtype = wtype

	def set_params(self,k,q,eta,beta):
		self.k = k
		self.q = q
		self.eta = eta
		self.beta = beta

	def set_wtype(self,wtype):
		assert wtype in wtype_list,\
		        "Weakness type {} is not supported. Supported {}".format(wtype,wtype_list)
		self.wtype = wtype

	def fit(self):
		y = [0]*self.n
		S = np.arange(self.n) # Initial set of indices
		r = int(np.ceil(self.k*self.eta))

		for i in xrange(k):
			# Phase 1
			if r >= len(S):
				r = len(S)
				print "Warning: sample size > remaining points"
			idx_Z = S[np.random.randint(0,len(S),r)]

			y_Z = self.clusterAssign(idx_Z)

			p = np.argmax([y_Z.count(t) for t in xrange(1,self.k+1)])+1
			idx_p = idx_Z[np.array(y_Z)==p]
			mpp = np.mean(self.X[:,idx_p],axis=1)

			# Phase 2
			idx_S_sorted = np.argsort(np.linalg.norm(
				                          self.X-np.tile(mpp,(self.n,1)).T,
				                          axis=0))
			idx_r = self.binarySearch(idx_S_sorted,S,idx_p,p)

			y[idx_S_sorted[:idx_r]] = i+1
			S = np.array(set(S)-set(idx_S_sorted[:idx_r]))

		self.y = y

	def clusterAssign(self,idx_Z):
		y_Z = [0]*len(idx_Z)
		k_max = 0
		for idx in idx_Z:
			y_Z[idx] = self.weakQuery()

		return y_Z

	def weakQuery(self,x,y):
		if self.wtype == "random":
			return 

		else:
			return 0

	def binarySearch(idx_S_sorted,S,idx_p,p):

		return rip