# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-05-05 20:22:13
# @Last Modified by:   twankim
# @Last Modified time: 2017-10-26 03:25:34

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def accuracy(y_true,y_pred):
    return 100*np.sum(y_true==y_pred)/float(len(y_true))

def mean_accuracy(y_true,y_pred):
    labels = np.unique(y_true)
    accuracy = np.zeros(len(labels))
    hamming = y_true==y_pred

    accuracy = [100*np.sum(hamming[y_true==label])/float(np.sum(y_true==label)) \
                for label in labels]
    return np.mean(accuracy)

def error(y_true,y_pred):
    return 100*np.sum(y_true!=y_pred)/float(len(y_true))

def mean_error(y_true,y_pred):
    labels = np.unique(y_true)
    num_error = np.zeros(len(labels))
    hamming = y_true!=y_pred

    error = [100*np.sum(hamming[y_true==label])/float(np.sum(y_true==label)) \
             for label in labels]
    return np.mean(error)

# Find best matching permutation of y_pred clustering
# Also need to change mpp of algorithm
def find_permutation(dataset,algo):
    # Calculate centers of original clustering
    label_org = list(np.unique(dataset.y))
    means_org = [np.mean(dataset.X[dataset.y==label,:],axis=0) for label in label_org]

    labels_map = {} # Map from algorithm's label to true label
    # Initialize label mapping
    for label in xrange(algo.k+1):
        labels_map[label] = 0

    if len(algo.labels)==0:
        return algo.y

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
def print_eval(eval_metric,res,etas,fname,
               is_sum=False,weak='random',params=None):
    assert weak in ['random','local','global'], \
                    "weak must be in ['random','local','global']"
    if weak == 'random':
        i_name = 'q'
        t_name = weak
    else:
        i_name = 'c_dist'
        t_name = weak +' distance'

    rep = res.shape[0]
    if not is_sum:
        df_res = pd.DataFrame(res.mean(axis=0),
                              columns=etas,
                              index=params
                              )
        df_res.index.name=i_name
        df_res.columns.name='eta'
        print "\n<{}. {}-weak (Averaged over {} experiments)>".format(
                eval_metric,t_name, rep)
    else:
        df_res = pd.DataFrame(res.sum(axis=0),
                              columns=etas,
                              index=params
                              )
        df_res.index.name=i_name
        df_res.columns.name='eta'
        print "\n<{}. {}-weak (Total Sum over {} experiments)>".format(
                eval_metric,t_name,rep)
    print df_res
    df_res.to_csv(fname)

# Plot eta v.s. evaluation
# res: rep x len(qs) x len(etas)
def plot_eval(eval_metric,res,etas,fig_name,
              is_sum=False,weak='random',params=None,res_org=None):
    assert weak in ['random','local','global'], \
                    "weak must be in ['random','local','global']"
    # cmap = plt.cm.get_cmap("jet", len(params)) -> cmap(i_p)
    cmap = ['g','r','b','k','y','m','c']

    if weak == 'random':
        i_name = 'q'
        t_name = weak
    else:
        i_name = 'c_{dist}'
        t_name = weak + ' distance'

    rep = res.shape[0]
    if not is_sum:
        res_plt = res.mean(axis=0)
        res_org_plt = res_org.mean(axis=0)
        f = plt.figure()
        plt.title(r"{}. {}-weak (Averaged over {} experiments)".format(
                            eval_metric,t_name,rep))
        for i_p,param in enumerate(params):
            plt.plot(etas,res_plt[i_p,:],
                     'x-',c=cmap[i_p],
                     label=r'SSAC(ours) ${}={}$'.format(i_name,param))
            if res_org is not None:
                plt.plot(etas,res_org_plt[i_p,:],
                         'o--',c=cmap[i_p],
                         label=r'SSAC(original) ${}={}$'.format(i_name,param))
        plt.xlabel(r"$\eta$ (Number of samples per cluster)")
        plt.ylabel(eval_metric)
    else:
        res_plt = res.sum(axis=0)
        res_org_plt = res_org.sum(axis=0)
        f = plt.figure()
        plt.title(r"{}. {}-weak (Total sum over {} experiments)".format(
                            eval_metric,t_name,rep))
        for i_p,param in enumerate(params):
            plt.plot(etas,res_plt[i_p,:],
                     'x-',c=cmap[i_p],
                     label=r'SSAC(ours) ${}={}$'.format(i_name,param))
            if res_org is not None:
                plt.plot(etas,res_org_plt[i_p,:],
                         'o--',c=cmap[i_p],
                         label=r'SSAC(oroginal) ${}={}$'.format(i_name,param))
        plt.xlabel(r"$\eta$ (Number of samples per cluster)")
        plt.ylabel(eval_metric)
    if "accuracy" in eval_metric.lower():
        plt.legend(loc=4)
        min_val = min(res_plt.min(),res_org_plt.min())
        max_val = max(res_plt.max(),res_org_plt.max())
        ylim_min = min_val-(max_val-min_val)*0.55
        ylim_max = max_val+(max_val-min_val)*0.05
    elif ("error" in eval_metric.lower()) or ("fail" in eval_metric.lower()):
        plt.legend(loc=1)
        max_val = max(res_plt.max(),res_org_plt.max())
        ylim_min = 0 - max_val*0.1
        ylim_max = max_val*1.35
    else:
        plt.legend(loc=4)
    plt.ylim([ylim_min,ylim_max])
    plt.xlim([0,np.round(1.2*max(etas))])
    
    
    f.savefig(fig_name,bbox_inches='tight')

def plot_hist(gammas,min_gamma,max_gamma,fig_name):
    rep = len(gammas)
    if rep>40:
        n_bins = int(rep/20)
    else:
        n_bins = 10
    f = plt.figure()
    plt.hist(gammas,normed=False,bins=n_bins)
    plt.title(r"Histogram of $\gamma$. min={}, max={} ({} generation)".format(min_gamma,max_gamma,rep))
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Number of data generations")

    f.savefig(fig_name,bbox_inches='tight')

def plot_cluster(X,y_true,y_pred,k,mpps,gamma,title,f_name,verbose,classes=None):
    if classes is not None:
        classes = classes
    else:
        classes = range(k+1)
    cmap = plt.cm.get_cmap("jet", k+1)
    if verbose:
        print " ... Plotting"
    f = plt.figure(figsize=(14,7))
    plt.suptitle(title)

    # Plot original clustering (k-means)
    plt.subplot(121)
    for i in xrange(1,k+1):
        idx = y_true==i
        plt.scatter(X[idx,0],X[idx,1],c=cmap(i),label=classes[i],alpha=0.7)
    # plt.scatter(X[:,0],X[:,1],c=y_true,label=classes)
    plt.title("True dataset ($\gamma$={:.2f})".format(gamma))
    plt.legend()

    # Plot SSAC result
    plt.subplot(122)
    for i in xrange(0,k+1):
        idx = np.array(y_pred)==i
        if sum(idx)>0:
            plt.scatter(X[idx,0],X[idx,1],c=cmap(i),label=classes[i],alpha=0.7)
    # plt.scatter(X[:,0],X[:,1],c=y_pred,label=classes)
    plt.title("SSAC result ($\gamma$={:.2f})".format(gamma))
    plt.legend()

    # Plot estimated cluster centers
    for t in xrange(k):
        mpp = mpps[t]
        plt.plot(mpp[0],mpp[1],'k^',ms=15,alpha=0.7)

    f.savefig(f_name,bbox_inches='tight')
    plt.close()
