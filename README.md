# bayesian-dropouts

Project for Bayesian statistics implementing the methods from [Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” Apr. 2016](https://arxiv.org/pdf/1506.02142.pdf)

## Data and code summary

We expiremented on one regression tasks and two classification tasks:
* regression : PM2.5 concentration in China, dataset available [here](https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities). The regression task is conducted both with the gaussian process and with the dropout NN models in [regression_GP.ipynb](https://github.com/strayMat/bayesian-dropouts/blob/master/regression_GP.ipynb) and [regression_NN.ipynb](https://github.com/strayMat/bayesian-dropouts/blob/master/regression_NN.ipynb).

* classification 1: MNIST (we used keras example datasets: **from keras.datasets import mnist**) implemented in [classification_MNIST.ipynb](https://github.com/strayMat/bayesian-dropouts/blob/master/clasification_MNIST.ipynb).

* classification 2: NotMnist, a similar dataset as mnist, slightly more complex on 10 characaters (available [here](http://yaroslavvb.com/upload/notMNIST/)). We downloaded the notMNIST_large.tar.gz (originally 600 000 samples) and cut it to 10000 images per classes using the file not_mnist_creation.py. The classification is implemented in [classification_notMnist.ipynb](https://github.com/strayMat/bayesian-dropouts/blob/master/classification_notMnist.ipynb).
 
Utils files implement the useful functions that we use for the regression and classification tasks.

### Instructions

* (1) explain the theoretical, computational and/or empirical methods, 
* (2) emphasize the main points of the paper, 
* (3) apply it to real data (that you will find). 

Bonus points will be considered if you are creative and add something insightful that is not in the original paper: this can be a theoretical point, an illustrative experiment, etc.

### Format

You can use either Python or R for the programming part. Please have each group send
* one report as a pdf (≤ 15 pages, with reasonable fonts and margins),
* one zipped folder containing your code and a detailed readme file with instructions to (compile/install and) run the code.

to all three teachers 1 no later than January 30th. There will be no deadline extension.
