# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import collections

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
        yh = h.evaluate(el[0])
        y = el[1]
        error += (yh - y)**2
    error /= len(x)
    return error


def normalize(x):
    for i in range(len(x[0])):
        x[:,i] -= np.amin(x[:,i])
        x[:,i] /= np.amax(x[:,i])
    return x


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
        self.w = np.zeros(deg ** dim)
        self.nodes = [np.linspace(0, 1, deg) for i in range(dim)]
        self.dist = self.nodes[0][1] - self.nodes[0][0]

    """
    Tansformation function (phi)
    @param x 1D-input value to transform
    @return phi(x)
    """
    def transform(self, x):

        out = 1
        if not isinstance(x, collections.Iterable):
            x = np.array([x])

        for i, dim in enumerate(x):
            phi = np.zeros(self.deg)
            for j in range(self.deg - 1):
                if self.nodes[i][j] <= dim and dim <= self.nodes[i][j+1]:
                    phi[j] = 1 - ((dim - self.nodes[i][j]) / self.dist)
                    phi[j+1] = 1 - phi[j]
                    break

            out = np.outer(out, np.array(phi)).flatten()

        return out

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
    def learn(self, X, y, l=None):
        if l is None:
            Xtilde = [self.transform(x) for x in X]
            Xdagger = np.linalg.pinv(Xtilde)
            self.w = np.dot(Xdagger, y)
        else:
            for i, x in enumerate(X):
                yh, Xdagger = self.evaluate(x)
                self.w += l.learn(Xdagger, y[i], yh)
        return self.w

    """
    Evaluate the polynomial approximator
    @param x Point to evaluate
    @return Predicted value
    """
    def evaluate(self, x):
        Xdagger = self.transform(x)
        return np.dot(self.w, Xdagger), Xdagger

    
if __name__ == "__main__":
    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
    # Shuffle data
    np.random.shuffle(data)
    # Seperate data in input and output values
    dataX = [el[0:4] for el in data]
    dataY = [el[4] for el in data]
    # Normalize input data
    dataX = normalize(np.asarray(dataX))
    # Split data in training and test set
    trainX = dataX[:6000]
    trainY = dataY[:6000]
    testX = dataX[6000:]
    testY = dataY[6000:]
    # Create hypothesis set
    h = GLT(10, 4)
    # Learn on the training data
    h.learn(trainX, trainY)
    # Display the in-sample-error
    print('In-Sample-Error:', calculateError(h, testX, testY))
