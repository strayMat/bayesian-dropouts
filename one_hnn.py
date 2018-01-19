
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras import regularizers
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

## In this file, we create a neural network with one hidden layer and apply it to the mnist dataset in order to address the classification problem.

# loading mnist and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[2]:


print(x_train.shape, y_train.shape, x_test.shape)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# network parameters
batch_size = 128
num_classes = 10
epochs = 20

# dimensions for the NN:
# input dimension
Q = x_train.shape[1]
# hidden layer dimension
K = 100
# output dimension : classification
D = num_classes

# dropout rate
p = 0.5
N = x_train.shape[0]
# l2 regularization
#prior length scale
l = 1e-2
# precision parameter, un peu au pif pour l'instant
tau = 1e-1

lambd = p*l**2/(2*N)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## Create a sequential model with Keras 

# In[3]:


model = Sequential()
model.add(Dense(K, input_shape = (Q, ), activation = 'relu', use_bias= True, kernel_regularizer = regularizers.l2(lambd),
               bias_regularizer = regularizers.l2(lambd)))
model.add(Dropout(p))
# softmax layer
model.add(Dense(num_classes, use_bias = False, activation = None, kernel_regularizer = regularizers.l2(lambd)))
model.add(Dropout(p))
model.add(Activation('softmax'))


# In[4]:


model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# In[5]:


# Training the network
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test))


# In[5]:


# Evaluation of the network
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ## Obtaining model uncertainty

# In[6]:


def my_relu(x):
    return x*(x>=0)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def softmax(x):
    norm = np.sum([np.exp(e) for e in x])
    return np.array([np.exp(e)/norm for e in x])

def softmax_in_out(x, model, nb_cl, T = 100) :
    W1, b, W2 = model.get_weights()
    S_in = np.zeros((T, nb_cl))
    S_out = np.zeros((T, nb_cl))
    
    for i in range(T):
        z1 = np.diag(np.random.rand(W1.shape[0]) > 0.5)
        z2 = np.diag(np.random.rand(W2.shape[0]) > 0.5)
        pred = np.dot(my_relu(np.dot(x, np.dot(z1, W1)) + b), np.dot(z2, W2))
        S_in[i] = pred
        S_out[i] = softmax(pred)
    return S_in, S_out

"return the k most predicted classes"
def best_preds(S, k):
    preds = np.unique(np.argmax(S_out, axis= 1), return_counts= True)
    classes = dict(zip(preds[1], preds[0]))
    best_classes = [classes[e] for e in np.sort(preds[1])[-3:]]
    return best_classes[::-1]

"return the dropout model prediction"
def dropout_pred(x, model, nb_cl, label = False):
    S_out = softmax_in_out(x, model, nb_cl)[1]
    pred = np.mean(S_out, axis = 0)
    if label == False:
        return(pred)
    else:
        return np.argmax(pred)    


# ## Vizualise data points

# In[7]:


# graphical functions to display the distribution of the predictions
"""
input:  S, a sample of predictions (T forward passes) of size Txnum_classes for an image
        classes, the classes for which displaying the probability distributions
output: The plot of the distributions of the predicted probabilities for the desired classes
"""

def plot_pred(S, classes):
    mapc = cm.tab10
    colors = dict(zip(classes, mapc.colors[:len(classes)]))
    #plt.figure(figsize= (8, 2))
    for cl in classes:
        plt.scatter(x = S[:,cl], y = np.zeros(S.shape[0]), label = cl, marker= '|', c = colors[cl], linewidth  = 2)
    for cl in classes:
        plt.scatter(x = np.mean(S[:,cl]), y = 0, c = colors[cl], linewidths= 4)
    plot = plt.legend()
    return plot

# Wrapper function for plotting the image and the distribution
def plot_pred_img(x, model, nb_class, S = 'softmax_out'):
    S_in, S_out = softmax_in_out(x, model, nb_cl=10)
    plt.subplot(121)
    if S == 'softmax_out':
        plot_pred(S_out, classes)
    elif S== 'softmax_in':
        plot_pred(S_in, classes)
    plt.subplot(122)
    x_pl = x.reshape((28,28))
    plt.imshow(x_pl, cmap = "gray")
    plt.show()
    return 
"""
Function to plot an image and its predicted class for several rotations of the image
input:  x, the input image
        m, the trained network model,
        nb_class, the number of prediction classes
        S, to plot the output or input of the softmax layer
output: the plot 
"""
def plot_rotated(x, model, nb_class,S = 'softmax_out'):
    for i in np.linspace()
    return
    


# In[10]:


np.linspace(0,180, 6)


# In[35]:


plot_pred_img(x_ex, model, nb_class=10)


# In[26]:


import imutils
ix = 1
x_ex = x_test[ix]

x_pl = x_ex.reshape((28,28))
x_plr = imutils.rotate(x_pl, 45)
plt.subplot(121)
plt.imshow(x_pl, cmap = "gray")
plt.subplot(122)
plt.imshow(x_plr, cmap = "gray")
plt.show()
print(x_plr.shape)
S_in, S_out = softmax_in_out(x_ex, model, nb_cl=10)
classes = best_preds(S_out, 3)
plot_pred(S_out, classes)
plt.show()


# In[137]:


ix = 1
x_ex = x_test[ix]
print("Network prediction: " + str(model.predict(x_ex.reshape((1, -1)))))
print("Networ label: "+ str(np.argmax(model.predict(x_ex.reshape((1, -1))))))

print("dropout prediction: "+ str(dropout_pred(x_ex, model, nb_cl=10)))
print("dropout label: "+ str(dropout_pred(x_ex, model, nb_cl=10, label=True)))

print("real label: " + str(np.argmax(y_test[ix])))

x_pl = x_ex.reshape((28,28))
plt.imshow(x_pl, cmap = "gray")
plt.show()

classes = best_preds(S_out, 3)
S_in, S_out = softmax_in_out(x_ex, model, nb_cl=10)
plot_pred(S_out, classes)
plt.show()

