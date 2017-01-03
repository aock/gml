import numpy as np
import sys
from ccpp_glt_linear import GLTlinear
from ccpp_glt_gaussian import GLTgauss
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
        error += (yh - y) ** 2
    error /= len(x)
    return error


def normalize(x):
    for i in range(len(x[0])):
        x[:, i] -= np.amin(x[:, i])
        x[:, i] /= np.amax(x[:, i])
    return x


def plot(x, y1, y2):
    plt.plot(x, y1, 'bo', x, y2, 'ro')
    plt.show()


if __name__ == "__main__":

    # Evaluate with Unknown Dataset
    if len(sys.argv) == 3:
        pocketfile = sys.argv[1]
        testdata = sys.argv[2]
        f = open(pocketfile, 'rb')
        pol_pocket = pickle.load(f)
        glt_pocket = pickle.load(f)
        f.close()
        data = np.genfromtxt(testdata, delimiter=",")

        # Seperate data in input and output values
        dataX = np.array([el[0:4] for el in data])
        dataY = np.array([el[4] for el in data])

        # normalize input values
        dataX = normalize(dataX)

        maxdim = 5

        print("--------------------------------")
        print("Polynome")

        for i in range(1, maxdim, 1):
            h = Polynomial(i)
            h.w = pol_pocket[i-1]
            eout = calculateError(h, dataX, dataY)
            print('Out-Sample-Error:  %d - %f' %
                  (i, eout))

        print("--------------------------------")
        print("GLT")
        # Create hypothesis set
        for i in range(2, maxdim, 1):
            h = GLT(i, 4)
            # Learn on the training data
            h.w = glt_pocket[i-2]
            # Display the in-sample-error
            eout = calculateError(h, dataX, dataY)
            print('Out-Sample-Error:  %d - %f' %
                  (i, eout))

        sys.exit()


    else:
        pol_pocket = [None, None, None, None]
        glt_pocket = [None, None, None]

    pol_err = [np.inf, np.inf, np.inf, np.inf]
    glt_err = [np.inf, np.inf, np.inf]
    glt_eout = []
    glt_ein = []
    pol_eout = []
    pol_ein = []


    times = 10
    for times in range(times):
        print(times)
        # Read data from file
        data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
        data2 = np.genfromtxt("ccpp_first_test.txt", delimiter=",")

        data = np.concatenate((data,data2))
        print("Number Trainingsets: " + str(len(data)))

        ratio = 1./5.


        # Seperate data in input and output values
        dataX = np.array([el[0:4] for el in data])
        dataY = np.array([el[4] for el in data])

        # Normalize input data
        dataX = normalize(np.asarray(dataX))
        for i,_ in enumerate(dataX):
            dataX[i][3] /= 10

        # Split data into train and test data
        split = int(len(data)*ratio)
        random_indices = np.arange(0, len(data))
        np.random.shuffle(random_indices)

        indices_check = list(random_indices[:split])
        indices_train = list(random_indices[split:])
        dataXTrain = dataX[indices_train]
        dataYTrain = dataY[indices_train]

        # OVERRIDE THEESE ARRAYS FOR OUT-OF-SAMPLE-ERROR
        dataXTest = dataX[indices_check]
        dataYTest = dataY[indices_check]
        if len(sys.argv) == 2 and not isinstance(eval(sys.argv[1]), int):
            print("Using " + str(sys.argv[1]) + " as test-data")
            dataTest = np.genfromtxt(sys.argv[1], delimiter=",")
            dataXTest = np.array([el[0:4] for el in dataTest])
            dataYTest = np.array([el[4] for el in dataTest])
            dataXTest = normalize(np.asarray(dataXTest))

        maxdim = 5

        eouts = []
        eins = []
        print("--------------------------------")
        print("Polynome")

        for i in range(1, maxdim, 1):
            h = Polynomial(i)
            h.learn(dataXTrain, dataYTrain)
            ein = calculateError(h, dataXTrain, dataYTrain)
            eout = calculateError(h, dataXTest, dataYTest)
            if pol_err[i -1 ] > eout:
                pol_err[i-1] = eout
                pol_pocket[i-1] = dc(h.w)
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
            if glt_err[i-2] > eout:
                glt_err[i-2] = eout
                glt_pocket[i-2] = dc(h.w)
            print('In-|Out-Sample-Error:  %d - %f | %f' %
                  (i, ein, eout))
            eouts.append(eout)
            eins.append(ein)
        glt_ein.append(dc(eins))
        glt_eout.append(dc(eouts))


    import pickle
    outfile = "weights"
    with open(outfile, 'wb') as f:
        #print(pol_pocket)
        #print(glt_pocket)
        pickle.dump(pol_pocket, f)
        pickle.dump(glt_pocket, f)


    if len(sys.argv) == 2:
        outfile = "data/data" + sys.argv[1]
        print("write data to " + outfile)
        with open(outfile, 'wb') as f:
            pickle.dump(pol_eout, f)
            pickle.dump(pol_ein, f)
            pickle.dump(glt_eout, f)
            pickle.dump(glt_ein, f)
