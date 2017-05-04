import numpy as np

class weakSSAC:
	def __init__(self,X,y,k,q=1,wtype="random"):
		self.X = X
		self.n, self.m = np.shape(X)
		self.y = y
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
		y = [0]*self.n
		S = np.arange(self.n) # Initial set of indices
		r = int(np.ceil(self.k*self.eta))

		self.mpps = []

		for i in xrange(self.k):
			# Phase 1
			if r >= len(S):
				r = len(S)
				print "Warning: sample size > remaining points"
			idx_Z = S[np.random.randint(0,len(S),r)]

			y_Z = self.clusterAssign(idx_Z)

			p = np.argmax([y_Z.count(t) for t in xrange(1,self.k+1)])+1
			idx_p = idx_Z[np.array(y_Z)==p]
			mpp = np.mean(self.X[idx_p,:],axis=0)
			# print "Size of Z_p: {}".format(len(idx_p))

			self.mpps.append(mpp)

			# Phase 2
			idx_S_sorted = S[np.argsort(np.linalg.norm(
				                          self.X[S,:]-np.tile(mpp,(len(S),1)),
				                          axis=1))]
			idx_radius = self.binarySearch(idx_S_sorted,idx_p)

			for i_assign in idx_S_sorted[:idx_radius]:
				y[i_assign] = i+1
			S = np.array(list(set(S)-set(idx_S_sorted[:idx_radius])))

		self.y = y

	def weakQuery(self,idx_x,idx_y):
		if self.wtype == "random":
			return np.random.binomial(1,self.q)*\
			       2*(int(self.y[idx_x]==self.y[idx_y])-0.5)
		else:
			print "Not sure!!!"
			return 0

	def clusterAssign(self,idx_Z):
		y_Z = [0]*len(idx_Z)
		k_max = 0
		for i,idx in enumerate(idx_Z):
			set_idx = [y_Z.index(k) for k in xrange(1,k_max+1)]
			if len(set_idx)>0:
				answers = [self.weakQuery(idx,idx_set) for idx_set in set_idx]
			else:
				answers = []

			if len(answers) == 0:
				y_Z[i] = k_max+1
				k_max += 1
			elif 1 in answers:
				y_Z[i] = y_Z[set_idx[answers.index(1)]]
			elif sum(answers) == -k_max:
				y_Z[i] = k_max+1
				k_max += 1
		return y_Z

	def binarySearch(self,idx_S_sorted,idx_p):
		idx_l = 0
		idx_r = len(idx_S_sorted)-1
		list_yes = list(idx_p)

		bs_num = 0
		while idx_l <= idx_r:
			idx_j = int(np.floor((idx_l+idx_r)*0.5))
			answer = self.weakQuery(idx_S_sorted[0],idx_S_sorted[idx_j])
			if answer == 1:
				idx_l = idx_j+1
				list_yes.append(idx_S_sorted[idx_j])
			elif answer == -1:
				idx_r = idx_j-1
			else:
				set_idx_B = np.random.randint(0,len(idx_p),self.beta-1)
				answers = [self.weakQuery(idx_p[idx_B],
					                      idx_S_sorted[idx_j]) for idx_B in set_idx_B]
				if 1 in answers:
					idx_l = idx_j+1
					list_yes.append(idx_S_sorted[idx_j])
				else:
					idx_r = idx_j-1
			bs_num += 1

		self.bs_num = bs_num

		# if idx_S_sorted[idx_l] in list_yes:
		# 	return idx_l+1
		# else:
		# 	if weakQuery(idx_[])
		# 	return idx_l
		return idx_l
