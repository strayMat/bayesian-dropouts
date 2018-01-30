import numpy as np

def my_relu(x):
    return x*(x>=0)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def softmax(x):
    norm = np.sum([np.exp(e) for e in x])
    return np.array([np.exp(e)/norm for e in x])


"""function to compute the T forward passes on the network and predict model uncertainty concerning a data point
Input:  x, datapoint for which we want to compute the prediction and the uncertainty
        model, trained neural network (assuming one hidden layer with dropout here)
        nb_cl, number of classes in the classification problem (ex: 10 for MNIST)
        T, number of forward passes (default is 100)
        drop_out, dropout rate
"""
def softmax_in_out(x, model, nb_cl, T = 100, drop_out = 0.5) :
    W1, b, W2 = model.get_weights()
    S_in = np.zeros((T, nb_cl))
    S_out = np.zeros((T, nb_cl))
    
    for i in range(T):
        z1 = np.diag(np.random.rand(W1.shape[0]) > drop_out)
        z2 = np.diag(np.random.rand(W2.shape[0]) > drop_out)
        pred = np.dot(my_relu(np.dot(x, np.dot(z1, W1)) + b), np.dot(z2, W2))
        S_in[i] = pred
        S_out[i] = softmax(pred)
    return S_in, S_out

"""return the dropout model prediction, given the model
"""
def dropout_pred(x, model, nb_cl, label = False, drop_out = 0.5):
    S_out = softmax_in_out(x, model, nb_cl, drop_out = drop_out)[1]
    pred = np.mean(S_out, axis = 0)
    if label == False:
        return(pred)
    else:
        return np.argmax(pred)    