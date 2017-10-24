#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `python tsne_embed.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
# Modifications copyright (C) 2017 UT Austin/Taewan Kim

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from data import DATASET

def Hbeta(D = np.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print "Computing pairwise distances..."
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print "Computing P-values for point ", i, " of ", n, "..."

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax =  np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy();
                if (betamax == np.inf) or (betamax == -np.inf):
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy();
                if (betamin == np.inf) or (betamin == -np.inf):
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta))
    return P


def pca(X = np.array([]), no_dims = 50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print "Preprocessing the data using PCA..."
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:,0:no_dims])
    return Y


def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float."
        return -1
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer."
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = max_iter
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4                                  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print "Iteration ", (iter + 1), ": error is ", C

        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    # Return solution
    return Y

def main(args):
    print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
    print "Running example on 2,500 MNIST digits..."

    X = np.loadtxt("dataset/mnist2500_X.txt")
    labels = np.loadtxt("dataset/mnist2500_labels.txt")

    if args.digits is not None:
        digits = [float(digit) for digit in args.digits.split(',')]
        print "Consider only digits :{}".format([int(digit) for digit in digits])

        # Select data with only selected labels
        idx = [(label in digits) for label in labels]
        labels = labels[idx]
        X = X[idx,:]
    else:
        digits = [float(digit) for digit in range(10)]

    Y = tsne(X, 2, 50, 20.0,args.max_iter)

    dict_label = {}
    temp_dict = {}
    for i,digit in enumerate(digits):
        dict_label[i] = digit
        temp_dict[digit] = i

    labels_k = np.array([temp_dict[label] for label in labels])

    mnist2500 = DATASET(Y, labels_k, X_org=X, dict_label=dict_label)

    with open('dataset/mnist2500.pkl','wb') as fp:
        pickle.dump(mnist2500, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if args.isplot:
        cmap = plt.cm.get_cmap("jet", len(digits))
        for i,digit in enumerate(digits):
            idx = labels==digit
            if sum(idx)>0:
                plt.scatter(Y[idx,0], Y[idx,1], c=cmap(i),label=int(digit),alpha=0.7)
        plt.title("t-SNE result ")
        plt.legend(loc='best')
        plt.show()

def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Run t-SNE on example MNIST data')
    parser.add_argument('-digits', dest='digits',
                        help='digits to consider',
                        default = None, type = str)
    parser.add_argument('-iter', dest='max_iter',
                        help='iter: Maximum number of interation',
                        default = 2000, type = int)
    parser.add_argument('-isplot', dest='isplot',
                        help='plot the result: True/False',
                        default = False, type = str2bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))
