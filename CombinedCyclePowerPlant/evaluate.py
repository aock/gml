import numpy as np
import sys
from ccpp_RLS import RLS
from ccpp_linear import GLT
from ccpp_polynomial import Polynomial
from matplotlib.pyplot import plot as plt
from copy import deepcopy as dc
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
    plt.plot(x, y1, 'bo', x, y2, 'ro')
    plt.show()


if __name__ == "__main__":

    pol_eout = []
    pol_ein = []
    glt_eout = []
    glt_ein = []


    for times in range(500):


        # Read data from file
        data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")



        #np.random.shuffle(data)
        dataX = []
        dataY = []

        split = 2000
        random_indices = np.arange(0,7000)
        np.random.shuffle(random_indices)

        indices_check = list(random_indices[:split])
        indices_train = list(random_indices[split:])

        # Seperate data in input and output values
        dataX = np.array([el[0:4] for el in data])
        dataY = np.array([el[4] for el in data])

        # Normalize input data
        dataX = normalize(np.asarray(dataX))


        dataXTrain  = dataX[indices_train]
        dataYTrain  = dataY[indices_train]
        dataXTest   = dataX[indices_check]
        dataYTest   = dataY[indices_check]

        maxdim = 5

        eouts = []
        eins = []
        print("Polynome")
        for i in range(1, maxdim, 1):
            h = Polynomial(i)
            h.learn(dataXTrain, dataYTrain)
            ein = calculateError(h, dataXTrain, dataYTrain)
            eout = calculateError(h, dataXTest, dataYTest)
            print('In-|Out-Sample-Error:  %d - %f | %f' %
            (i, ein, eout))
            eouts.append(eout)
            eins.append(ein)
        pol_ein.append(dc(eins))
        pol_eout.append(dc(eouts))


        eouts = []
        eins = []
        print("--------------------------------")
        print("GLT")
        # Create hypothesis set
        for i in range(2, maxdim, 1):
            h = GLT(i, 4)
            # Learn on the training data
            h.learn(dataXTrain, dataYTrain)
            # Display the in-sample-error
            ein = calculateError(h, dataXTrain, dataYTrain)
            eout = calculateError(h, dataXTest, dataYTest)
            print('In-|Out-Sample-Error:  %d - %f | %f' %
            (i, ein, eout))
            eouts.append(eout)
            eins.append(ein)
        glt_ein.append(dc(eins))
        glt_eout.append(dc(eouts))


    import pickle
    outfile = sys.argv[1]
    with open(outfile, 'wb') as f:
        pickle.dump(pol_eout, f)
        pickle.dump(pol_ein, f)
        pickle.dump(glt_eout, f)
        pickle.dump(glt_ein, f)



    """
    print("--------------------------------")
    print("RLS")


    for i in range(2,maxdim,1):
        pocket = None
        error = np.inf
        h = GLT(i, 4)
        l = RLS(i ** 4, 1)
        _i = 0
        # Learn on the training data
        #for _i in range(100):
        while True and _i < 1000:
            h.learn(dataXTrain, dataYTrain, l)
            err = calculateError(h, dataXTest, dataYTest)
            if error > err and abs(error - err) > 0.01:
                error = err
                pocket = dc(h.w)
            else:
                print("Fehler: ", err)
                break
            print("%d - Best: %f | Current: %f" % (_i, error, err))
            _i += 1

        if pocket is not None:
            h.w = pocket

            # Display the in-sample-error
            print('In-|Out-Sample-Error:  %d - %f | %f' %
            (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))
        else:
            print('In-|Out-Sample-Error:  %d - keine Approximation gefunden' % i)
    """
