
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

from lib_classifNN import *
from lib_visu_classif import *

from tqdm import tqdm
"""In this file, we create a neural network with one hidden layer and apply it to the not_mnist dataset in order to address the classification problem and computing uncertainty as developped in  Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” Apr. 2016.
"""

path = 'data/notmnist_train.npy'


# In[9]:


#loading data
data = np.load(path)
#dictionnary to swap between numerical label and letter
letter2label = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], np.arange(10)))
label2letter = dict(zip(np.arange(10), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']))


# In[3]:


sep = 80000
x_train = data.tolist()["x_train"][:sep]
x_test = data.tolist()["x_train"][sep:]

y_train = data.tolist()["y_train"][:sep]
y_test = data.tolist()["y_train"][sep:]


# In[4]:


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
print("----------Data loaded and preprocessed----------")


# In[5]:


# Create the neural network model and train it
model = Sequential()
model.add(Dense(K, input_shape = (Q, ), activation = 'relu', use_bias= True, kernel_regularizer = regularizers.l2(lambd), bias_regularizer = regularizers.l2(lambd)))
model.add(Dropout(p))
# softmax layer
model.add(Dense(num_classes, use_bias = False, activation = None, kernel_regularizer = regularizers.l2(lambd)))
model.add(Dropout(p))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

print("----------Model built---------")


# In[6]:


print("----------Training the model...---------")
# Training the network
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test))

# Evaluation of the network
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[8]:


def plot_rotated(x, model, nb_class, S = 'softmax_out', label2name = None):
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
            plot_pred(S_out, classes, yp = -i, legend=l, label2name = label2name)
        elif S == 'softmax_in':
            plot_pred(S_in, classes, yp = -i, legend=l, label2name = label2name)
    for (i, x) in zip(np.arange(nb_rotations), x_rotated):
        plt.subplot(gs[i, -1])
        plt.imshow(x, cmap = 'gray')
    # remove uninteresting axes
    plt.setp([b.get_yticklabels() for b in fig.axes], visible = False)
    plt.setp([b.get_xticklabels() for b in fig.axes[1:]], visible = False)
    plt.show()
    return


# In[13]:


# Rough comparison between the classic neural network and the bayesian estimation
ix = 8
x_ex = x_test[ix]
print("Network prediction: " + str(model.predict(x_ex.reshape((1, -1)))))
print("Networ label: "+ str(np.argmax(model.predict(x_ex.reshape((1, -1))))))

print("dropout prediction: "+ str(dropout_pred(x_ex, model, nb_cl=10)))
print("dropout label: "+ str(dropout_pred(x_ex, model, nb_cl=10, label=True)))

print("real label: " + str(np.argmax(y_test[ix])))

x_pl = x_ex.reshape((28,28))
plt.imshow(x_pl, cmap = "gray")
plt.show()

classes = np.argsort(model.predict(x_ex.reshape(1,-1))[0])[-3:]
S_in, S_out = softmax_in_out(x_ex, model, nb_cl=10)
plot_pred(S_out, classes, legend=True)
plt.show()


print("----------Assessing uncertainty for rotated example of a data point---------")

ix = 15411
x_ex = x_test[ix]
plot_rotated(x_ex, model , nb_class=10, label2name= label2letter)
plt.show()

