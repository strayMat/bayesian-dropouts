import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import imutils

import numpy as np 

from lib_classifNN import *

# graphical functions to display the distribution of the predictions for the classification case 
# The most useful function to display the final result is plot_rotated

"""
input:  S, a sample of predictions (T forward passes) of size Txnum_classes for an image
        classes, the classes for which displaying the probability distributions
output: The plot of the distributions of the predicted probabilities for the desired classes
"""
def plot_pred(S, classes, yp = 0, legend = False):
    mapc = cm.tab10
    colors = dict(zip(classes, mapc.colors[:len(classes)]))
    #plt.figure(figsize= (8, 2))
    for cl in classes:
        plt.scatter(x = S[:,cl], y = np.array([1]*S.shape[0])*yp, label = cl, marker= '|', c = colors[cl], linewidth  = 2)
    for cl in classes:
        plt.scatter(x = np.mean(S[:,cl]), y = yp, c = colors[cl], linewidths= 4)
    if legend:plt.legend()
    return

# Wrapper function for plotting the image and the distribution, [deprecated in favor of plot_rotated]
def plot_pred_img(x, model, nb_class, S = 'softmax_out', yp = 0):
    classes = np.argsort(model.predict(x_ex.reshape(1,-1))[0])[-3:]
    S_in, S_out = softmax_in_out(x, model, nb_cl=10)
    fig = plt.figure(figsize = (10,2))
    gs =gridspec.GridSpec(1,2, width_ratios= [7,1])
    a = plt.subplot(gs[0])
    if S == 'softmax_out':
        plot_pred(S_out, classes, yp, legend = True)
    elif S == 'softmax_in':
        plot_pred(S_in, classes, yp, legend = True)
    b = plt.subplot(gs[1])
    x_pl = x.reshape((28,28))
    plt.imshow(x_pl, cmap = "gray")
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.setp([b.get_xticklabels(), b.get_yticklabels(), a.get_yticklabels()], visible = False)
    return 

"""
Function to plot an image and its predicted classes for several rotations of the image
input:  x, the input image
        model, the trained neural network model,
        nb_class, the number of prediction classes
        S, to plot the output or input of the softmax layer (either 'softmax_in' or 'softmax_out')
output: the plot 
"""
def plot_rotated(x, model, nb_class, S = 'softmax_out'):
    classes = np.argsort(model.predict(x.reshape(1,-1))[0])[-3:]
    nb_rotations = 6 # change this if more or less rotated versions of the image is needed
    x_pl = x.reshape((28,28))
    x_rotated = [imutils.rotate(x_pl, r) for r in np.linspace(0,180, nb_rotations)]
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(nb_rotations, nb_rotations)
    ax1 = plt.subplot(gs[:,:-1])
    for (i, x) in zip((np.arange(nb_rotations)), x_rotated):        
        S_in, S_out = softmax_in_out(x.reshape((1,784)), model, nb_cl=nb_class)
        # trick to show legend only once 
        l = False
        if i == 0:l = True
        if S == 'softmax_out':
            plot_pred(S_out, classes, yp = -i, legend=l)
        elif S == 'softmax_in':
            plot_pred(S_in, classes, yp = -i, legend=l)
    for (i, x) in zip(np.arange(nb_rotations), x_rotated):
        plt.subplot(gs[i, -1])
        plt.imshow(x, cmap = 'gray')
    # remove uninteresting axes
    plt.setp([b.get_yticklabels() for b in fig.axes], visible = False)
    plt.setp([b.get_xticklabels() for b in fig.axes[1:]], visible = False)
    plt.show()
    return