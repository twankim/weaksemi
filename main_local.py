# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-02-24 17:46:51
# @Last Modified by:   twankim
# @Last Modified time: 2017-10-22 01:06:57

import numpy as np
import time
import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ssac import weakSSAC
from gen_data import genData
from utils import *

weak = "local"
delta = 0.99
base_dir= os.path.join('./results',weak)

def main(args):
    rep = args.rep
    k = args.k
    n = args.n
    m = args.m
    std = args.std
    # qs = [float(q) for q in args.qs.split(',')]
    etas = [float(eta) for eta in args.etas.split(',')]
    beta = args.beta
    i_plot = np.random.randint(0,rep) # Index of experiment to plot the figure
    verbose = args.verbose

    cs = [float(q) for q in args.cs.split(',')]
    
    res_acc = np.zeros((rep,len(cs),len(etas))) # Accuracy of clustering
    res_mean_acc = np.zeros((rep,len(cs),len(etas))) # Mean accuracy of clustering (per cluster)
    # res_err = np.zeros((rep,len(qs),len(etas))) # Number of misclustered points
    res_fail = np.zeros((rep,len(cs),len(etas))) # Number of Failure
    gammas = np.zeros(rep)
    nus = np.zeros((rep,len(cs)))
    rhos = np.zeros((rep,len(cs)))

    # Make directories to save results
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    res_dir = base_dir + '/{}_{}'.format(args.min_gamma,args.max_gamma)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for i_rep in xrange(rep):
        # Generate Synthetic data
        # m dimensional, n points, k cluster
        # min_gamma: minimum gamma margin
        if verbose:
            print "({}/{})... Generating data".format(i_rep+1,rep)
        dataset = genData(n,m,k,args.min_gamma,args.max_gamma,std)
        X,y_true,ris = dataset.gen()
        gamma = dataset.gamma
        gammas[i_rep] = gamma
        print "({}/{})... Synthetic data is generated: gamma={}, (n,m,k,std)=({},{},{},{})".format(
                i_rep+1,rep,gamma,n,m,k,std)

        algo = weakSSAC(X,y_true,k,wtype=weak,ris=ris)
        # Test SSAC algorithm for different c's and eta's (fix beta in this case)
        # Calculate proper eta and beta based on parameters including delta
        for i_c,c_dist in enumerate(cs):
            assert (c_dist>0.5) & (c_dist<=1.0), "c_dist must be in (0.5,1]"

            nus[i_rep,i_c] = float(gamma) + (1-c_dist)
            rhos[i_rep,i_c] = c_dist

            if verbose:
                print "   - Proper eta={}, beta={} (delta={})".format(
                        dataset.calc_eta(delta,weak=weak,nu=nus[i_rep,i_c],rho=rhos[i_rep,i_c]),
                        dataset.calc_beta(delta,weak=weak,nu=nus[i_rep,i_c],rho=rhos[i_rep,i_c]),
                        delta)

            for i_eta,eta in enumerate(etas):
                if verbose:
                    print "     <Test: c_dist={}, eta={}, beta={}>".format(c_dist,eta,beta)
                algo.set_params(eta,beta,rho=rhos[i_rep,i_c],nu=nus[i_rep,i_c])

                if not algo.fit():
                    # Algorithm has failed
                    res_fail[i_rep,i_c,i_eta] = 1
                    i_plot = np.random.randint(i_rep+1,rep) # Index of experiment to plot the figure
                
                y_pred = algo.y
                mpps = algo.mpps # Estimated cluster centers
                # print "     ... Clustering is done. Number of binary search steps = {}\n".format(algo.bs_num)

                # For evaluation & plotting, find best permutation of cluster assignment
                y_pred_perm = find_permutation(dataset,algo)

                # Calculate accuracy and mean accuracy
                res_acc[i_rep,i_c,i_eta] = accuracy(y_true,y_pred_perm)
                res_mean_acc[i_rep,i_c,i_eta] = mean_accuracy(y_true,y_pred_perm)

                # # Calculate number of errors
                # res_err[i_rep,i_c,i_eta] = error(y_true,y_pred_perm)

                if (i_rep == i_plot) and (m<=2):
                    title = r"SSAC with {} weak oracle ($\eta={}, \beta={}, \nu={:.2f}, \rho={:.2f}$)".format(
                                weak,eta,beta,nus[i_rep,i_c],rhos[i_rep,i_c])
                    f_name = res_dir+'/fig_n{}_m{}_k{}_c{:03d}_e{:d}.png'.format(n,m,k,int(100*c_dist),int(eta))
                    plot_cluster(X,y_true,y_pred_perm,k,mpps,gamma,
                                 title,f_name,verbose)

    # Write result as table
    fname = res_dir+'/res_{}_n{}_m{}_k{}.csv'.format("acc",n,m,k)
    print_eval("Accuracy(%)",res_acc,etas,
               res_dir+'/res_{}_n{}_m{}_k{}.csv'.format("acc",n,m,k),weak=weak,params=cs)
    print_eval("Mean Accuracy(%)",res_mean_acc,etas,
               res_dir+'/res_{}_n{}_m{}_k{}.csv'.format("meanacc",n,m,k),weak=weak,params=cs)
    # print_eval("# Error(%)",res_err,qs,etas,
    #            res_dir+'/res_{}_n{}_m{}_k{}.csv'.format("err",n,m,k))
    print_eval("# Failure",res_fail,etas,
               res_dir+'/res_{}_n{}_m{}_k{}.csv'.format("fail",n,m,k),
               is_sum=True,weak=weak,params=cs)

    # if args.isplot:
    # Plot Accuracy vs. eta
    fig_name = res_dir+'/fig_{}_n{}_m{}_k{}.pdf'.format("acc",n,m,k)
    plot_eval("Accuracy(%)",res_acc,etas,fig_name,weak=weak,params=cs)

    # Plot Mean Accuracy vs. eta
    fig_name = res_dir+'/fig_{}_n{}_m{}_k{}.pdf'.format("meanacc",n,m,k)
    plot_eval("Mean Accuracy(%)",res_mean_acc,etas,fig_name,weak=weak,params=cs)

    # Plot Failure vs. eta
    fig_name = res_dir+'/fig_{}_n{}_m{}_k{}.pdf'.format("fail",n,m,k)
    plot_eval("# Failure",res_fail,etas,fig_name,is_sum=True,weak=weak,params=cs)

    # Plot histogram of gammas
    fig_name = res_dir+'/fig_gamma_hist.pdf'
    plot_hist(gammas,args.min_gamma,args.max_gamma,fig_name)

    if args.isplot:
        plt.show()
    
def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Test Semi-Supervised Active Clustering with Weak Oracles: Random-weak model')
    parser.add_argument('-rep', dest='rep',
                        help='Number of experiments to repeat',
                        default = 10000, type = int)
    parser.add_argument('-k', dest='k',
                        help='Number of clusters in synthetic data',
                        default = 3, type = int)
    parser.add_argument('-n', dest='n',
                        help='Number of data points in synthetic data',
                        default = 600, type = int)
    parser.add_argument('-m', dest='m',
                        help='Dimension of data points in synthetic data',
                        default = 2, type = int)
    parser.add_argument('-std', dest='std',
                        help='standard deviation of Gaussian distribution (default:1.5)',
                        default = 2.25, type = float)
    parser.add_argument('-qs', dest='qs',
                        help='Probabilities q (not-sure with 1-q) ex) 0.7,0.85,1',
                        default = '0.7,0.85,1', type = str)
    parser.add_argument('-etas', dest='etas',
                        help='etas: parameter for sampling (phase 1) ex) 10,50',
                        default = '2,5,10,20,30', type = str)
    parser.add_argument('-beta', dest='beta',
                        help='beta: parameter for sampling (phase 2)',
                        default = 1, type = int)
    parser.add_argument('-g_min', dest='min_gamma',
                        help='minimum gamma margin (default:1)',
                        default = 1.0, type = float)
    parser.add_argument('-g_max', dest='max_gamma',
                        help='minimum gamma margin (default:1)',
                        default = 1.1, type = float)
    parser.add_argument('-cs', dest='cs',
                        help='Fractions to set distance-weak parameters (0.5,1] ex) 0.7,0.85,1',
                        default = '0.6,0.8,1', type = str)
    parser.add_argument('-isplot', dest='isplot',
                        help='plot the result: True/False',
                        default = False, type = str2bool)
    parser.add_argument('-verbose', dest='verbose',
                        help='verbose: True/False',
                        default = False, type = str2bool)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))
