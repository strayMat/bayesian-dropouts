import numpy as np
from tqdm import tqdm


def gradient_descent_dropout(x, y, p1, p2, ep=0.0001, max_iter=1000, alpha = 0.0001, decrease_lr = False):

    res = []
    converged = False
    iter = 0
    N = x.shape[0] # number of samples
    Q = x.shape[1]
    D = y.shape[1]
    K = 100


    # initial weight matrices
    M1 = np.random.normal(0., 1.,(Q, K))
    M2 = np.random.normal(0., 1., (K, D))
    m = np.random.normal(0., 1., (1,K))
    L_GP_MC = 0

    tau = p1/(2*0.01*N)

    for it in tqdm(range(max_iter), desc="main loop"):

        z1v = np.zeros((Q,N))
        z2v = np.zeros((K,N))

        for i in range(N):
            z1v[:,i] = np.random.binomial(size=Q, n=1, p=p1)
            z2v[:,i] = np.random.binomial(size=K, n=1, p=p2)

        grad_M1 = np.zeros((Q,K))
        grad_M2 = np.zeros((K,D))
        grad_m = np.zeros((1,K))

        L_tmp = L_GP_MC
        L_GP_MC = - p1*np.linalg.norm(M1)**2 - p2*np.linalg.norm(M2)**2 - np.linalg.norm(m)**2

        for i in range(N):
            #Bernoulli vectors for dropouts
            z1 = np.diag(z1v[:,i])
            z2 = np.diag(z2v[:,i])

            # Computing prediction for the x
            y_pred = (1/K)**(1/2)*np.matmul(np.maximum(0,(np.matmul(x[i],np.matmul(z1,M1))+m)),np.matmul(z2,M2))
            ynew =  y_pred - y[i]
            L_GP_MC -= tau*np.linalg.norm(ynew)**2

            partial_Y_tild = 2*(y_pred - y[i])*tau

            z1 = np.diag(z1v[:,i])
            z2 = np.diag(z2v[:,i])

            # Computing Gradient for each parameters

            grad_M2_tmp = np.dot(np.transpose(np.maximum(0, np.dot(np.matrix(x[i]), np.dot(z1, M1)) + m)), np.matrix(partial_Y_tild))
            grad_M2 += np.dot(z2, grad_M2_tmp)

            grad_m1 = (np.dot(np.matrix(x[i]), np.dot(z1, M1)) + m > 0)
            grad_m2 = (np.dot(np.matrix(partial_Y_tild), np.transpose(np.dot(z2, M2))))
            grad_m += np.multiply(grad_m1, grad_m2)

            grad_M11 = np.transpose(np.matrix(x[i]))
            grad_M12 = np.multiply(np.dot(np.matrix(partial_Y_tild), np.transpose(np.dot(z2, M2))), (np.dot(np.matrix(x[i]), np.dot(z1, M1)) + m > 0))
            grad_M1 += np.dot(z1, np.dot(grad_M11, grad_M12))


        grad_M1 += p1 * M1
        grad_M2 += p2 * M2
        grad_m += m

        # Using decreasing step
        if(decrease_lr):
            alpha_tmp = alpha/(1.+(0.0001 * iter))**(0.25)
        else:
            alpha_tmp = alpha

        # update M
        M1 -= alpha_tmp * grad_M1
        M2 -= alpha_tmp * grad_M2
        m -= alpha_tmp * grad_m

        res.append(L_GP_MC)

        if(np.abs(L_tmp - L_GP_MC) < ep):
            print("break")
            return res, M1, M2, m

    return res, M1, M2, m

class Gaussian_Process:

    def __init__(self, p1=0.8, p2=0.9, hidden_units=100):
        self.p1=p1
        self.p2=p2
        self.K=hidden_units

        self.M1 = 0
        self.M2 = 0
        self.m = 0

    def fit(self, X_train, Y_train, p1=0.8, p2=0.9, ep=0.0001, max_iter=1000, alpha = 0.0005, verbose=False):

        history_loss, self.M1, self.M2, self.m = gradient_descent_dropout(X_train, Y_train, p1, p2, ep=1e-7, max_iter=max_iter, alpha = alpha)
        return history_loss

    def predict(self, X_test, p1=0.8, p2=0.9):

        Q = self.M1.shape[0]
        K = self.K
        N = X_test.shape[0] # number of samples
        if(X_test.shape[1]!= Q):
            print("ERREUR DE DIM")


        Y_predict = []
        for i in range(N):
            z1 = np.diag(np.random.binomial(size=Q, n=1, p=p1))
            z2 = np.diag(np.random.binomial(size=K, n=1, p=p2))

            # Computing prediction for the x
            y_pred = (1/K)**(1/2)*np.matmul(np.maximum(0,(np.matmul(X_test[i],np.matmul(z1,self.M1))+self.m)),
                np.matmul(z2,self.M2))
            Y_predict.append(y_pred)
        return Y_predict

    def accuracy(self, X_test, Y_test, p1 = 0.8, p2=0.9):
        Y_predict = self.predict(X_test, p1, p2)
        acc = (np.linalg.norm(Y_predict - Y_test)**2)/(np.linalg.norm(Y_test)**2)
        return acc

    def plot_acc(self, X_test, Y_test, p1=0.8, p2=0.9):
        Y_predict = self.predict(X_test, p1, p2)
        plt.plot(Y_predict, Y_test)
        plt.show()

