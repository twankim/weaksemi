import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from data import DATASET
from sklearn.manifold import TSNE

def main(args):
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

    tsne = TSNE(n_components=2,
                init='pca',
                random_state=0,
                n_iter=args.max_iter,
                perplexity=args.p,
                verbose=args.verbose,
                learning_rate=200,
                early_exaggeration=15.0)

    X_tsne = tsne.fit_transform(X)

    dict_label = {}
    temp_dict = {}
    for i,digit in enumerate(digits):
        dict_label[i] = digit
        temp_dict[digit] = i

    labels_k = np.array([temp_dict[label] for label in labels])

    mnist2500 = DATASET(X_tsne, labels_k, X_org=X, dict_label=dict_label)

    with open('dataset/mnist2500.pkl','wb') as fp:
        pickle.dump(mnist2500, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if args.isplot:
        f = plt.figure()
        cmap = plt.cm.get_cmap("jet", len(digits))
        for i,digit in enumerate(digits):
            idx = labels==digit
            if sum(idx)>0:
                plt.scatter(X_tsne[idx,0], X_tsne[idx,1],
                c=cmap(i),label=int(digit),alpha=0.7)
        plt.title(r"t-SNE result ($\gamma={}$)".format(mnist2500.gamma))
        plt.legend(loc='best')
        plt.show()
        f.savefig('dataset/cluster.png', bbox_inches='tight')

def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Run t-SNE on example MNIST data')
    parser.add_argument('-digits', dest='digits',
                        help='digits to consider',
                        default = None, type = str)
    parser.add_argument('-init_d', dest='init_d',
                        help='Initial dimension',
                        default = 50, type = int)
    parser.add_argument('-p', dest='p',
                        help='perplexity',
                        default = 30.0, type = float)
    parser.add_argument('-iter', dest='max_iter',
                        help='iter: Maximum number of interation',
                        default = 2000, type = int)
    parser.add_argument('-isplot', dest='isplot',
                        help='plot the result: True/False',
                        default = False, type = str2bool)
    parser.add_argument('-verbose', dest='verbose',
                        help='verbosity level of tsne. default:0',
                        default = 0, type = int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))
