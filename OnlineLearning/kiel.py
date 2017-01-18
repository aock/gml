from __future__ import division
import os, sys, numpy as np
import matplotlib.pyplot as plt
from OLLib import *
import collections

# 2008
dio8 = np.array([[i*7, 1+i*7, 2+i*7] for i in range(53)]).flatten()
fr8  = np.array([3+i*7 for i in range(53)])
sa8  = np.array([4+i*7 for i in range(53)])
so8 = np.array([5+i*7 for i in range(53)])
mo8  = np.array([6+i*7 for i in range(53)])
ff8  = np.array([1,81,84,122,133,143,228,277,306,360,361])
sof8 = np.concatenate((ff8,so8))
np.sort(sof8)

mo8 = np.array([x for x in mo8 if x not in ff8])
dio8 = np.array([x for x in dio8 if x not in ff8])
fr8 = np.array([x for x in fr8 if x not in ff8])
sa8 = np.array([x for x in sa8 if x not in ff8])

for arr in [mo8, dio8, fr8, sa8, sof8]:
    for i,e in enumerate(arr):
        if e > 59:
            arr[i] -= 1

mo8 = mo8[np.where(mo8 < 365)]
dio8 = dio8[np.where(dio8 < 365)]
fr8 = fr8[np.where(fr8 < 365)]
sa8 = sa8[np.where(sa8 < 365)]
sof8 = sof8[np.where(sof8 < 365)]

# 2009
dio9 = np.array([[i*7, 5+i*7, 6+i*7] for i in range(53)]).flatten()
fr9  = np.array([1+i*7 for i in range(53)])
sa9  = np.array([2+i*7 for i in range(53)])
so9 = np.array([3+i*7 for i in range(53)])
mo9  = np.array([4+i*7 for i in range(53)])
ff9  = np.array([1,6,100,103,121,141,152,162,227,276,359,360])
sof9 = np.concatenate((ff9,so9))
np.sort(sof9)

mo9 = np.array([x for x in mo9 if x not in ff9])
dio9 = np.array([x for x in dio9 if x not in ff9])
fr9 = np.array([x for x in fr9 if x not in ff9])
sa9 = np.array([x for x in sa9 if x not in ff9])

mo9 = mo9[np.where(mo9 < 365)]
dio9 = dio9[np.where(dio9 < 365)]
fr9 = fr9[np.where(fr9 < 365)]
sa9 = sa9[np.where(sa9 < 365)]
sof9 = sof9[np.where(sof9 < 365)]


"""
This class contains a linear regressor on a GLT-approximator
for 1D problems
"""
class GLTgauss:
    """
    Constructor
    The nodes are implicity placed equidistantly on [-1;1]
    For a general solution neither the input interval of [-1;1] nor the
    equidistant positioning should be assumed
    @param deg Number of nodes
    """
    def __init__(self, deg, xmin=0, xmax=1, dim=1, pos=None):
        self.deg = deg
        self.w = np.zeros(deg ** dim)
        self.nodes = [pos if pos is not None else np.linspace(xmin, xmax, deg)]
        self.dist = self.nodes[0][1] - self.nodes[0][0]
        self.scale = 1 / (2*np.sqrt(-np.log(1/2)))


    def gauss(self, x, u, sigma):
        return np.exp(-(x-u)**2 / (2*sigma**2))

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
            for j in range(self.deg):
                 phi[j] = self.gauss(dim, self.nodes[i][j], self.dist * self.scale)
            phi = phi / np.sum(phi)
            out = np.outer(out, np.array(phi)).flatten()

        return out

#        for i in range(len(x)):
#            phi = np.zeros(self.deg)
#            for j in range(self.deg - 1):
#                if self.nodes[i][j] <= x[i] and x[i] <= self.nodes[i][j+1]:
#                    phi[j] = 1 - ((x[i] - self.nodes[i][j]) / self.dist)
#                    phi[j+1] = 1 - phi[j]
#                    break
#            out.append(phi)
#        return np.hstack(out)

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
        return np.dot(self.w, Xdagger)#, Xdagger


def batch(data, approx):
    Xtilde = [approx.transform(x) for x in data[:,1]]
    Xdagger = np.linalg.pinv(Xtilde)
#    data[:,2] += 1
    approx.w = np.dot(Xdagger, data[:,2])

def learn(data, approximator, learner):
    for i, x in enumerate(data):
        learner.learn(x[1], x[2])


def jorotheEval(first, second):
    first[:, 1] += 10
    first[:, 1] /= 20
    second[:, 1] += 10
    second[:, 1] /= 20

    print(min(first[:,1]), max(first[:,1]), min(first[:,2]), max(first[:,2]))
    #plt.plot(range(96*7), first[:96*7,2])
    #approx = GLT(97, 0, 1)
    #batch(first, approx)


    #for d in range(50):
    #    plt.plot(range(96), first[7*d*96:7*d*96+96,2])
        #plt.plot(range(96), second[(d+2)*96:(d+3)*96,2])
    #    plt.ylim([-1,1])
    #    plt.show()


    firstSetSize= 7 # First 2 Months for Init
    dim = 3
    aprox = Polynomial
    learner = PALearner
    # GLT Optimum: GLT(99) 6706
    # Pol Optimum: Pol(20 5746         Pol(1) -> 6090
    # Leg Optimum: Leg(3) 6329
    # Che Optimum: Che(3)
    # GGT Optimum: GGT(99) 5552

    a_mo  = aprox(dim)
    a_dio = aprox(dim)
    a_fr  = aprox(dim)
    a_sa  = aprox(dim)
    a_sof = aprox(dim)


    def concat(arr, data, d):
        if arr is None:
            arr = data[d*96:(d+1)*96]
        else:
            arr = np.concatenate((arr, data[d*96:(d+1)*96]))
        return arr

    data_mo = None
    data_dio = None
    data_fr = None
    data_sa = None
    data_sof = None
    for d in range(365):
        if d in mo8:
            data_mo = concat(data_mo, first, d)
        elif d in dio8:
            data_dio = concat(data_dio, first, d)
        elif d in fr8:
            data_fr = concat(data_fr, first, d)
        elif d in sa8:
            data_sa = concat(data_sa, first, d)
        else:
            data_sof = concat(data_sof, first, d)

    batch(data_mo[0:firstSetSize*96], a_mo)
    learner_mo = learner(a_mo)

    batch(data_dio[0:firstSetSize*96], a_dio)
    learner_dio = learner(a_dio)

    batch(data_fr[0:firstSetSize*96], a_fr)
    learner_fr = learner(a_fr)

    batch(data_sa[0:firstSetSize*96], a_sa)
    learner_sa = learner(a_sa)

    batch(data_sof[0:firstSetSize*96], a_sof)
    learner_sof = learner(a_sof)

    print("Learned, start predicting")

    def predictLearn(aprox, learner, data, d, last, plot=False):
        p = []
        t = np.zeros(96)
        for m in range(96):
            p.append(aprox.evaluate(m/96))
        p = np.array(p)
        #print(p)
        #p += last - p[0]
        last = p[-1]
        #print(d)
        data2 = data[d*96:(d+1)*96]
        #np.random.shuffle(data)
        ran = np.linspace(0,95,96)
        #np.random.shuffle(ran)
        for m in ran:
            m = int(m)
            datum = data2[m]
            t[m] = (datum[2])
            #datum[2] += -data[d*96, 2] + p[0]
            learner.learn(datum[1], datum[2])
        t = np.array(t)
        if plot:
            plt.ylim([-1,1])
            plt.plot(range(96), p)
            plt.plot(range(96), t)
            plt.show()
        return p, t, last, np.sum(np.abs(p - t))


    #plt.ion()
    last = 0
    cumLoss = 0
    cL = []
    for d in range(365):
        print("Day: ", d)
        if d in mo9:
            _, _, last, loss = predictLearn(a_mo, learner_mo, second, d, last)
            cumLoss += loss
        elif d in dio9:
            _, _, last, loss = predictLearn(a_dio, learner_dio, second, d, last)
            cumLoss += loss
        elif d in fr9:
            _, _, last, loss = predictLearn(a_fr, learner_fr, second, d, last)
            cumLoss += loss
        elif d in sa9:
            _, _, last, loss = predictLearn(a_sa, learner_sa, second, d, last)
            cumLoss += loss
        else:
            _, _, last, loss = predictLearn(a_sof, learner_sof, second, d, last)
            cumLoss += loss
        cL.append(cumLoss)

    plt.plot(range(365), cL)
    plt.show()

    print("Fehler: ", cumLoss)



if __name__ == "__main__":
    np.random.seed(12345)
    first = np.fromfile('firstYear.dat', sep=' ').reshape([35040, 3])
    second = np.fromfile('secondYear.dat', sep=' ').reshape([35040, 3])

    jorotheEval(first, second)
