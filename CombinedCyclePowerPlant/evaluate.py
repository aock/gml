import numpy as np
import sys
from ccpp_linear import GLT
from ccpp_polynomial import Polynomial
from matplotlib.pyplot import
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


def plot(x, y1, y2):
    plt.plot(x, y2, 'bo', x, y1, 'ro')
    plt.show()


if __name__ == "__main__":

    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")

    #np.random.shuffle(data)

    # Seperate data in input and output values
    dataX = [el[0:4] for el in data]
    dataY = [el[4] for el in data]

    # Normalize input data
    dataX = normalize(np.asarray(dataX))

    saveValues = 5000

    dataXTrain  = dataX[:saveValues]
    dataYTrain  = dataY[:saveValues]
    dataXTest   = dataX[saveValues:]
    dataYTest   = dataY[saveValues:]

    print("Polynome")
    for i in range(4, 16, 1):
        h = Polynomial(i)
        h.learn(dataXTrain, dataYTrain)
        print('In-|Out-Sample-Error:  %d - %f | %f' %
        (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))


    print("--------------------------------")
    print("GLT")
    # Create hypothesis set
    for i in range(18, 22, 1):
        h = GLT(i, 4)
        # Learn on the training data
        h.learn(dataXTrain, dataYTrain)
        # Display the in-sample-error
        print('In-|Out-Sample-Error:  %d - %f | %f' %
        (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))