import numpy as np
import sys
from ccpp_RLS import RLS
from ccpp_linear import GLT
from ccpp_polynomial import Polynomial
from matplotlib.pyplot import plot as plt
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


def normalize(x):
    for i in range(len(x[0])):
        x[:,i] -= np.amin(x[:,i])
        x[:,i] /= np.amax(x[:,i])
    return x


def plot(x, y1, y2):
    plt.plot(x, y2, 'bo', x, y1, 'ro')
    plt.show()


if __name__ == "__main__":

    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")

    #np.random.shuffle(data)
    dataX = []
    dataY = []

    random_indices = np.arange(0,7000)
    np.random.shuffle(random_indices)

    indices_check = list(random_indices[:2000])
    indices_train = list(random_indices[2000:])

    # Seperate data in input and output values
    dataX = np.array([el[0:4] for el in data])
    dataY = np.array([el[4] for el in data])

    # Normalize input data
    dataX = normalize(np.asarray(dataX))

    saveValues = 5000

    dataXTrain  = dataX[indices_train]
    dataYTrain  = dataY[indices_train]
    dataXTest   = dataX[indices_check]
    dataYTest   = dataY[indices_check]

    print("Polynome")
    for i in range(4, 16, 1):
        h = Polynomial(i)
        h.learn(dataXTrain, dataYTrain)
        print('In-|Out-Sample-Error:  %d - %f | %f' %
        (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))


    print("--------------------------------")
    print("GLT")
    # Create hypothesis set
    h = GLT(20, 4)
    # Learn on the training data
    h.learn(dataXTrain, dataYTrain)
    # Display the in-sample-error
    print('In-|Out-Sample-Error:  %d - %f | %f' %
        (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))



    print("--------------------------------")
    print("RLS")
    for i in range(18, 22, 1):
        h = GLT(i, 4)
        l = RLS(i * 4, 1)
        # Learn on the training data
        h.learn(dataXTrain, dataYTrain, l)
        # Display the in-sample-error
        print('In-|Out-Sample-Error:  %d - %f | %f' %
        (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))
