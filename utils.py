# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 20:22:13
# @Last Modified by:   twankim
# @Last Modified time: 2017-05-08 22:25:56

import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_true,y_pred):
	return np.sum(y_true==y_pred)/float(len(y_true))

def mean_accuracy(y_true,y_pred):
	labels = np.unique(y_true)
	accuracy = np.zeros(len(labels))
	hamming = y_true==y_pred

	accuracy = [np.sum(hamming[y_true==label])/float(np.sum(y_true==label))\
	            for label in labels]
	return np.mean(accuracy)

# Find best matching permutation of y_pred clustering
# Also need to change mpp of algorithm
def find_permutation(dataset,algo):
	# Calculate centers of original clustering
	label_org = list(np.unique(dataset.y))
	means_org = [np.mean(dataset.X[dataset.y==label,:],axis=0) for label in label_org]

	labels_map = {} # Map from algorithm's label to true label
	labels_map[0] = 0
	for label,mpp in zip(algo.labels,algo.mpps):
		# Calculate distance between estimated center and true centers
		dist = [np.linalg.norm(mpp-mean_org) for mean_org in means_org]
		
		# Assign true cluster label to the algorithm's label
		idx_best = np.argmin(dist)
		labels_map[label] = label_org[idx_best]
		
		# Remove assigned label from the list
		del means_org[idx_best]
		del label_org[idx_best]

	return [labels_map[y] for y in algo.y]

# Plot eta v.s. evaluation
# res: rep x len(qs) x len(etas)
def plot_eval(eval_metric,res,qs,etas):
	rep = res.shape[0]
	plt.figure()
	plt.title(r"{} of SSAC (# of experiments={})".format(eval_metric,rep))
	for i_q,q in enumerate(qs):
		plt.plot(etas,res.mean(axis=0)[i_q,:],'x-',label=r'$q={}$'.format(q))
	plt.xlabel(r"$\eta$")
	plt.ylabel(eval_metric)
	plt.ylim([0,1])
	plt.xlim([0,np.round(1.2*max(etas))])
	plt.legend(loc=4)
