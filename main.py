# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-02-24 17:46:51
# @Last Modified by:   twankim
# @Last Modified time: 2017-05-04 11:26:06

import numpy as np
import time
import sys
import argparse
import matplotlib.pyplot as plt

from ssac import weakSSAC
from gen_data import genData

weak = "randaom"

def main(args):
	k = args.k
	n = args.n
	m = args.m
	qs = [float(q) for q in args.qs.split(',')]
	etas = [float(eta) for eta in args.etas.split(',')]
	beta = args.beta

	# Generate Synthetic data
	# m dimensional, n points, k cluster
	# min_gamma: minimum gamma margin
	print "... Generating data"
	dataset = genData(n,m,k,args.min_gamma,1)
	X,y_true = dataset.gen()
	print "... Synthetic data is generated: gamma={}, (n,m,k)=({},{},{})".format(
		                                            dataset.gamma,n,m,k)

	for q in qs:
		for eta in etas:
			print "<Test: n={}, m={}, q={}, eta={}, beta={}>".format(n,m,q,eta,beta)
			algo = weakSSAC(X,y_true,k,q)
			algo.set_params(eta,beta)
			algo.fit()
			y_pred = algo.y
			mpps = algo.mpps
			print "... Clustering is done. Binary Search number = {}".format(algo.bs_num)
	
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
	
	if args.isplot:
		plt.show()

def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Test Semi-Supervised Active Clustering with Weak Oracles: Random-weak model')
    parser.add_argument('-k', dest='k',
                        help='Number of clusters in synthetic data',
                        default = 3, type = int)
    parser.add_argument('-n', dest='n',
                        help='Number of data points in synthetic data',
                        default = 100000, type = int)
    parser.add_argument('-m', dest='m',
                        help='Dimension of data points in synthetic data',
                        default = 2, type = int)
    parser.add_argument('-qs', dest='qs',
                        help='Probabilities q (not-sure with 1-q) ex) 0.7,0.85,1',
                        default = '0.7,0.85,1', type = str)
    parser.add_argument('-etas', dest='etas',
                        help='etas: parameter for sampling (phase 1) ex) 30,100',
                        default = '30,100', type = str)
    parser.add_argument('-beta', dest='beta',
                        help='beta: parameter for sampling (phase 2)',
                        default = 10, type = int)
    parser.add_argument('-gamma', dest='min_gamma',
                        help='minimum gamma margin (default:1)',
                        default = 1, type = int)
    parser.add_argument('-isplot', dest='isplot',
                        help='plot the result',
                        default = True, type = str2bool)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_args()
	print "Called with args:"
	print args
	sys.exit(main(args))
