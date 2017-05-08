# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 20:22:13
# @Last Modified by:   twankim
# @Last Modified time: 2017-05-06 23:40:12

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

def find_permutation(y_true,algo):
	return True
	# Find best matching permutation of y_pred clustering
	# Also need to change mpp of algorithm