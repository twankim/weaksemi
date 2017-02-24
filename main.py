import numpy as np
import time
from ssac import weakSSAC
from gen_data import genData

k = 3
n = 1000
# qs = [0.7,0.9,1]
qs = [1]
eta = 20
beta = 10


m = 2
gen_data.genData(n,m,k)
X,y_true = gen_data.gen()

for q in qs:
	algo = ssac.weakSSAC(X,k,q)
	algo.set_params(eta,beta)
	algo.fit
	y_pred = algo.y
