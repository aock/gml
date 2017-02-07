from sklearn.svm import SVC
import numpy as np
from copy import deepcopy as dc
from time import time
import pickle
from numpy.polynomial.legendre import legval as leg
import scipy.special as sp
import sys
import matplotlib.pyplot as plt

def createLegrendeParams(d):
    d += 1
    return [1 / d] * d


def plot(z):
    x = np.arange(0.1, 10.1, 0.1)
    y = np.arange(1, 21, 1)
    #z = np.reshape(z, (len(y),len(x)))
    xx, yy = np.meshgrid(x, y)
    plt.pcolor(xx, yy, z)
    plt.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    plt.xlabel("C")
    plt.ylabel("$\mathregular{Q_f}$")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    #np.random.seed(1337)

    iterations = 100
    n = 200
    max_d = 21

    error = np.zeros((max_d -1, iterations))

    for i, C in enumerate(np.arange(0.1, 10.1, 0.1)):
        print(C)
        for d in range(1, max_d):

            e = 0
            for k in range(iterations):
                svm = SVC(C=C, kernel='rbf', gamma='auto')

                ve = True
                while ve:
                    x1 = np.random.rand(n) * 2 - 1
                    x2 = np.random.rand(n) * 2 - 1
                    y = leg(x1, createLegrendeParams(d))
                    label = np.zeros(n)
                    label[x2 > y] = 1
                    try:
                        svm.fit(np.vstack([x1, x2]).T, label)
                        ve = False
                        e += len(svm.support_vectors_)
                    except ValueError:
                        pass
            error[d-1, i] = e / iterations / n

    plot(error)