# Semi-Supervised Active Clustering with Weak Oracles (WeakSemi)
## Overview
We provide a novel and efficient semi-supervised active clustering algorithms for center-based clustering task, which can discover the inherent clustering of an imperfect oracle. Our work is motivated by the [SSAC algorithm](https://papers.nips.cc/paper/6449-clustering-with-same-cluster-queries.pdf), and the following question: “Is it possible to perform a clustering task efficiently even with a non-ideal domain expert?”. We answer this question by formulating different types of weak oracles and prove that the SSAC algorithm can still work well under uncertainties by using properly modified binary search schemes.

Here, we provide an implementation of our weak SSAC algorithm with an option of using random-weak model. Synthetic data is generated using Guassian distribution with several options. The algorithm is guaranteed to recover a ground truth clustering of the data with high probability. Please read our [paper(Need to be updated)](http://sites.google.com/a/utexas.edu/twankim) for details. We implemented the unified version of weak SSAC algorithm which can handle both random-weak and distance-weak oracles.

## How to run our code
You will run the main.py file with several options.
-rep "Number of experiments to repeat"
-k "Number of clusters in synthetic data"
-n "Number of data points in synthetic data"
-m "Dimension of data points in synthetic data"
-std "Standard deviation of Guassian distribution in generating data"
-qs "Probabilities q to test. Oracle says not-sure with probability at most 1-q. ex) 0.7,0.85,1"
-etas "Parameters for sampling in Phase 1 of the algorithm. ex) 10,50"
-beta "Parameter for sampling in Phase 2 of the algorithm. ex) 10"
-g_min "Minimum gamma margin for generating data"
-g_max "Maximum gamma margin for generating data"
-isplot "Plot the result True/False"
-verbose "True/False"
ex)
```
python main.py -k 3 -qs 0.7,0.85,1 -etas 2,5,10,50 -g_min 1.0 -g_max 1.2
```
