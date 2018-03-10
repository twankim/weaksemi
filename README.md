# Semi-Supervised Active Clustering with Weak Oracles (WeakSemi)
## Overview
We provide a novel and efficient semi-supervised active clustering algorithms for center-based clustering task, which can discover the inherent clustering of an imperfect oracle. Our work is motivated by the [SSAC algorithm](https://papers.nips.cc/paper/6449-clustering-with-same-cluster-queries.pdf), and the following question: “Is it possible to perform a clustering task efficiently even with a non-ideal domain expert?”. We answer this question by formulating different types of weak oracles and prove that the SSAC algorithm can still work well under uncertainties by using properly modified binary search schemes.

- **Weak Same-cluster Query**: Are these two points in the same cluster?
- **Answer**
  - *Yes*
  - *No*
  - *Not-sure*

Here, we provide an implementation of our weak SSAC algorithm with an option of using **Random or Local/Global Distance-Weak Oracle Models**, i.e. answers “*not-sure*” randomly with some fixed probability or based on distance between given points. Synthetic data is generated using Guassian distribution with several options. The algorithm is guaranteed to recover a ground truth clustering of the data with high probability. Please read our [paper](https://arxiv.org/abs/1709.03202) for details. We implemented the unified version of weak SSAC algorithm which can handle both random-weak and distance-weak oracles.

Shorter version of the paper, [Relaxed Oracles for Semi-Supervised Clustering](https://arxiv.org/abs/1711.07433), was presented at [NIPS 2017 Workshop: Learning with Limited Labeled Data: Weak Supervision and Beyond (LLD 2017)](https://lld-workshop.github.io/papers/LLD_2017_paper_19.pdf).

## How to run our code
You will run the main.py file with several options.
```
-rep "Number of experiments to repeat"

-k "Number of clusters in synthetic data"

-n "Number of data points in synthetic data"

-m "Dimension of data points in synthetic data"

-std "Standard deviation of Guassian distribution in generating data"

-qs "Probabilities q to test. Oracle says not-sure with probability at most 1-q. ex) 0.7,0.85,1"

-cs 'Fractions to set distance-weak parameters (0.5,1] ex) 0.7,0.85,1'

-etas "Parameters for sampling in Phase 1 of the algorithm. ex) 10,50"

-beta "Parameter for sampling in Phase 2 of the algorithm. ex) 10"

-g_min "Minimum gamma margin for generating data"

-g_max "Maximum gamma margin for generating data"

-isplot "Plot the result True/False"

-verbose "True/False"
```

ex) run main_local.py or main_global.py for distance-weak oracle models
```
python main.py -k 3 -qs 0.7,0.85,1 -etas 2,5,10,50 -g_min 1.0 -g_max 1.2
```

## Comparison of weakSSAC and SSAC
run *compare_{weakness}.py* to compare original SSAC and our improved weakSSAC. Original SSAC will receive random answers (in the same cluster/not in the same cluster with probability 0.5) whenever it encounters not-sure situation.
