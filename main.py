import numpy as np
import time
from ssac import weakSSAC
from gen_data import genData
import matplotlib.pyplot as plt

k = 3
n = 10000
qs = [0.7,0.85,1]
# qs = [1]
etas = [30,100]
beta = 10
weak = "randaom"


m = 2
dataset = genData(n,m,k,1)
X,y_true = dataset.gen()

for q in qs:
	for eta in etas:
		algo = weakSSAC(X,y_true,k,q)
		algo.set_params(eta,beta)
		algo.fit()
		y_pred = algo.y
		mpps = algo.mpps

		gamma = dataset.gamma

		plt.figure(figsize=(14,7))
		plt.suptitle(r"SSAC with {} weak oracle ($q={},\eta={}, \beta={}$)".format(weak,q,eta,beta))

		plt.subplot(121)
		plt.scatter(X[:,0],X[:,1],c=y_true)
		plt.title("True dataset ($\gamma$={:.2f})".format(gamma))

		plt.subplot(122)
		plt.scatter(X[:,0],X[:,1],c=y_pred)
		plt.title("SSAC result ($\gamma$={:.2f})".format(gamma))

		for t in xrange(k):
			mpp = mpps[t]
			plt.plot(mpp[0],mpp[1],'g^',ms=10)

plt.show()