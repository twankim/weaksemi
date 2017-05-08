# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 20:22:13
# @Last Modified by:   twankim
# @Last Modified time: 2017-05-08 18:23:11

import numpy as np

def accuracy(y_true,y_pred):
	return np.sum(y_true==y_pred)/float(len(y_true))

def mean_accuracy(y_true,y_pred):
	labels = np.unique(y_pred)
	accuracy = np.zeros(len(labels))
	hamming = y_true==y_pred

	accuracy = [np.sum(hamming[y_true==label])/float(np.sum(y_true==label))\
	            for label in labels]
	return np.mean(accuracy)

# Find best matching permutation of y_pred clustering
# Also need to change mpp of algorithm
def find_permutation(dataset,algo):
	label_org = np.unique(dataset.y)
	means_org = [np.mean(dataset.X[dataset.y==label,:],axis=0) for label in label_org]
	labels_best = [] # Best label
	for mpp in algo.mpps:
		# Calculate distance between estimated center and true centers
		dist = [np.norm(mpp-mean_org) for mean_org in means_org]
		idx_best = np.argmin(dist)
		
		labels_best.append(algo.labels[idx_best])

	return [labels_best[algo.labels.index(y)] for y in algo.y]