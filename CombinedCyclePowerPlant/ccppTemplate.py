# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

"""
Calcuate the MSE on a given dataset
@param h The hypothesis to evaluate
@param x The input values of the test data
@param y The output value of the test data
@return The mean squared error
"""
def calculateError(h, x, y):
    error = 0
    for el in zip(x, y):
        yh, _ = h.evaluate(el[0])
        y = el[1]
        error += (yh - y)**2
    error /= len(x)
    return error
    
def calculateErrorSVR(svr, x, y):
    error = 0
    
    yh = svr.predict(x)
    for i,el in enumerate(yh):
        error += (el-y[i])**2
    error /= len(x)
    return error
    


def normalize(x):
    for i in range(len(x[0])):
        x[:,i] -= np.amin(x[:,i])
        x[:,i] /= np.amax(x[:,i])
    return x

def plot(x, y1, y2):
    plt.plot(x, y2, 'bo', x, y1, 'ro')
    plt.show()


"""
This class contains a linear regressor on a GLT-approximator
for 1D problems
"""
class GLT:
    """
    Constructor
    The nodes are implicity placed equidistantly on [-1;1]
    For a general solution neither the input interval of [-1;1] nor the
    equidistant positioning should be assumed
    @param deg Number of nodes
    """
    def __init__(self, deg, dim):
        self.deg = deg
        self.w = np.zeros(deg * dim)
        self.nodes = [np.linspace(0, 1, deg) for i in range(dim)]
        self.dist = self.nodes[0][1] - self.nodes[0][0]

    """
    Tansformation function (phi)
    @param x 1D-input value to transform
    @return phi(x)
    """
    def transform(self, x):
        out = []
        for i in range(len(x)):
            phi = np.zeros(self.deg)
            for j in range(self.deg - 1):
                if self.nodes[i][j] <= x[i] and x[i] <= self.nodes[i][j+1]:
                    phi[j] = 1 - ((x[i] - self.nodes[i][j]) / self.dist)
                    phi[j+1] = 1 - phi[j]
                    break
            out.append(phi)
        return np.hstack(out)

    """
    Performs the learning of the approximator
    @param X Matrix (Vector for 1D) containing all x-input values
    @param y Vector of corresponding output values
    """
    def learn(self, X, y, l):
        for i, x in enumerate(X):
            yh, Xdagger = self.evaluate(x)
            self.w += l.learn(Xdagger, y[i], yh)

    """
    Evaluate the polynomial approximator
    @param x Point to evaluate
    @return Predicted value
    """
    def evaluate(self, x):
        Xdagger = self.transform(x)
        return np.dot(self.w, Xdagger), Xdagger


class RLS:
    def __init__(self, dim, forgF):
        # covariance matrix
        self.P = np.identity(dim)
        # forgetting factor
        self.r = forgF

    def learn(self, x, y, yh):
        w = np.dot(self.P, x) * (y - yh) / (self.r + np.inner(x, np.dot(self.P, x)))
        self.P = self.P / self.r - np.outer(np.dot(self.P, x), np.dot(x, self.P)) \
                                   / (self.r * (self.r + np.inner(np.dot(x, self.P), x)))
        return w

    
if __name__ == "__main__":
    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
    np.random.shuffle(data)
    # Seperate data in input and output values
    
    

    dataX = [list( np.array(el)[[0,1,2,3]]) for el in data]
    
    dataY = [el[4] for el in data]
    
    
    # Normalize input data
    dataX = normalize(np.asarray(dataX))
    
    num_train_data = 1000
    
    trainX = dataX[:num_train_data]
    trainY = dataY[:num_train_data]
    testX = dataX[num_train_data:]
    testY = dataY[num_train_data:]
    
    
    
    #svr_lin = SVR(kernel='linear', C=1e3)  
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    #y_lin = svr_lin.fit(dataX, dataY).predict(dataX)
    svr_rbf.fit(trainX, trainY)
    y_in_sample_rbf = svr_rbf.predict(trainX)
    y_out_of_sample_rbf = svr_rbf.predict(testX)
    
    print('In-Sample-Error SVR RBF:', calculateErrorSVR(svr_rbf, trainX, trainY) )
    print('Out-Of-Sample-Error SVR RBF:', calculateErrorSVR(svr_rbf, testX, testY) )    
    
    lw = 2
    #plt.scatter([x[0] for x in dataX], dataY, color='darkorange', label='data')
    #plt.plot(dataX, y_lin, color='c', lw=lw, label='Linear model')
    #plt.plot(dataX, y_rbf, 'bo', lw=lw, label='RBF model')
    #plt.plot([x[0] for x in dataX], y_poly, 'bo', lw=lw, label='Polynomial model')
    
    #plt.show()
    
    #print('In-Sample-Error SVR Linear:', calculateErrorSVR(svr_lin, dataX, dataY) )
    
    
    # Create hypothesis set
    #h = GLT(100, len(dataX[0]))
    # Initialize learning algorithm
    #l = RLS(100 * len(dataX[0]), 1)
    # Learn on the training data
    #h.learn(dataX, dataY, l)
    # Display the in-sample-error
    #print('In-Sample-Error:', calculateError(h, dataX, dataY))
    #plot(dataX, [h.evaluate(i)[0] for i in dataX], dataY)