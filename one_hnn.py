
# coding: utf-8

# In[15]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop

## In this file, we create a neural network with one hidden layer and apply it to the mnist dataset in order to address the classification problem.

# loading mnist and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[16]:


print(x_train.shape, y_train.shape, x_test.shape)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# dimensions for the NN:
# input dimension
Q = x_train.shape[1]
# hidden layer dimension
K = 100
# output dimension 
D = 512

# network parameters
batch_size = 128
num_classes = 10
epochs = 20


# create a sequential model
model = Sequential()
model.add(Dense(K, input_shape = (Q, ), activation = 'relu'))
model.add(Dropout(0.5))
# output layer
model.add(Dense(D, input_shape = (K, )))
model.add(Dropout(0.5))
# softmax layer
model.add(Dense(num_classes, activation = 'softmax'))


# In[17]:


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# In[18]:


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


# In[19]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

