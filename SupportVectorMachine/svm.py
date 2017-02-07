from __future__ import division
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


def target(x, deg):
    return sp.legendre(deg)(x)

def plotScatter(z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0.1, 10.1, 0.1)
    y = np.arange(1, 21, 1)
    xs, ys = np.meshgrid(x, y)
    ax.scatter(xs.flatten(), ys.flatten(), z)
    plt.show()

def plot(z):
    x = np.arange(0.1, 10.1, 0.1)
    y = np.arange(1, 21, 1)
    z = np.reshape(z, (len(y),len(x)))
    xx, yy = np.meshgrid(x, y)
    plt.pcolor(xx, yy, z)
    plt.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    plt.xlabel("C")
    plt.ylabel("$\mathregular{Q_f}$")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    np.random.seed(12345)
    result = []
    for deg in np.arange(1, 21, 1):
        for X in np.arange(0.1, 10.1, 0.1):
            error = 0
            for i in range(100):
                inputs = np.random.rand(200, 2) * 2 - 1
                classes = np.array([np.sign(x[1] - target(x[0], deg)) for x in inputs])
                svm = SVC(C=X, kernel='rbf', gamma='auto')
                svm.fit(inputs, classes)
                error += len(svm.support_vectors_) / 200
            result.append(error / 100)
        print deg
    plot(np.asarray(result))
