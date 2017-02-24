import numpy as np
import time
from ssac import weakSSAC
from gen_data import genData
import matplotlib.pyplot as plt

k = 3
n = 1000
# qs = [0.7,0.9,1]
qs = [1]
eta = 50
beta = 1


m = 2
dataset = genData(n,m,k,0.7)
X,y_true = dataset.gen()

for q in qs:
	algo = weakSSAC(X,y_true,k,q)
	algo.set_params(eta,beta)
	algo.fit()
	y_pred = algo.y
	mpps = algo.mpps

	gamma = dataset.gamma

	plt.figure(figsize=(14,7))
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
