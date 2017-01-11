# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:52:01 2016

@author: Jonas Schneider
"""

from __future__ import division

import numpy as np

from matplotlib import pyplot as plt
from OLLib import GLT
from OLLib import Polynomial
from OLLib import PALearner
from OLLib import RLSLearner


def generateConstant(n, noise=0, param=None):
    if type(n) is np.ndarray:
        return [1] * len(n)
    else:
        x = np.random.rand(n) * 2 - 1
        y = [1] * n + np.random.rand(n) * 2 * noise - noise
        return x, y


def generateLinear(n, noise=0, param=None):
    if type(n) is np.ndarray:
        return n * 2
    else:
        x = np.random.rand(n) * 2 - 1
        y = x * 2 + np.random.rand(n) * 2 * noise - noise
        return x, y


def generateSine(n, noise=0, param=None):
    if type(n) is np.ndarray:
        return np.sin(4 * np.pi * n)
    else:
        x = np.random.rand(n) * 2 - 1
        y = np.sin(4 * np.pi * x) + np.random.rand(n) * 2 * noise - noise
        return x, y


def generateStep(n, noise=0, param=None):
    if type(n) is np.ndarray:
        return np.sign(n)
    x = np.random.rand(n) * 2 - 1
    y = np.sign(x)
    return x, y


def generateComplex(n, noise=0, param=None):
    if type(n) is np.ndarray:
        return (1 / (0.1 + n**2)) + 0.05 * np.cos(100 * n)
    x = np.random.rand(n) * 2 - 1
    y = (1 / (0.1 + x**2)) + 0.05 * np.cos(100 * x) + np.random.rand(n) * 2 * noise - noise
    return x, y


def generateShift(n, noise=0, param=None):
    if type(n) is np.ndarray:
        if param["t"] < param["len"] / 2:
            return np.sin(4 * np.pi * n)
        else:
            return np.cos(4 * np.pi * n)
    x = np.random.rand(n) * 2 - 1
    y = np.append(np.sin(4 * np.pi * x[:int(n/2)]), np.cos(4 * np.pi * x[int(n/2):]))
    return x, y


def generateDrift(n, noise=0, param=None):
    if type(n) is np.ndarray:
        if param["t"] < param["driftStart"]:
            return np.sin(4 * np.pi * n)
        elif param["t"] > param["driftEnd"]:
            return np.sin(5 * np.pi * n)
        else:
            return np.sin((4 + (param["t"] - param["driftStart"]) / (param["driftEnd"] - param["driftStart"])) * np.pi * n)
    x = np.random.rand(n) * 2 - 1
    y = np.zeros(n)
    for i in range(n):
        if i < param["driftStart"]:
            y[i] = np.sin(4 * np.pi * x[i])
        elif i > param["driftEnd"]:
            y[i] = np.sin(5 * np.pi * x[i])
        else:
            y[i] = np.sin((4 + (n-param["driftStart"]) / (param["driftEnd"] - param["driftStart"])) * np.pi * x[i])
    return x, y


def calculateGTloss(targetFunc, approximator, funcParam=None, numPoints=1000):
    gtX = np.linspace(-1, 1, numPoints)
    gtY = targetFunc(gtX, param=funcParam, noise=0)
    evalY = np.zeros(numPoints)
    for i, x in enumerate(gtX):
        evalY[i] = approximator.evaluate(x)
    return 1 / numPoints * np.sum(np.power(np.subtract(gtY, evalY), 2))


def learnAndPlot(targetFunc, numPoints, approximator, degree, learner, funcParam=None, noise=0, sort=False, resolution=1000, learnParam=None, plot=False, gTLoss=False):
    # Create data set to learn from
    gtX = np.linspace(-1, 1, resolution)
    gtY = targetFunc(gtX, noise=0, param=funcParam)
    dataX, dataY = targetFunc(numPoints, noise=noise, param=funcParam)
    batch = zip(dataX, dataY)
    if sort:
        batch = sorted(batch)
    approx = approximator(degree)
    learner = learner(approx, learnParam)

    if gTLoss:
        gtLoss = np.zeros(numPoints)
        gtLoss[0] = calculateGTloss(targetFunc, approx, funcParam)
    cumLoss = np.zeros(numPoints)

    evalY = np.zeros(resolution)
    # GUI code from here on
    if plot:
        plt.figure()
        plt.ion()
        for j, x in enumerate(gtX):
            evalY[j] = approx.evaluate(x)
        plt.subplot(311)
        plt.xlim([-1,1])
        tfPts, = plt.plot(gtX, gtY, linewidth=2, color='k')
        approxPts, = plt.plot(gtX, evalY, linewidth=2)
        ptPts = plt.scatter(dataX[0], dataY[0], color='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.subplot(313)
        gtPts, = plt.plot(gtLoss[0], linewidth=2)
        plt.xlim([1,numPoints])
        plt.ylim([0,gtLoss[0]*2])
        plt.xlabel('Training Steps')
        plt.ylabel('Ground Truth Loss')
        plt.subplot(312)
        cumLossPts, = plt.plot(cumLoss[0], linewidth=2)
        plt.xlim([1,numPoints])
        plt.ylim([0,50])
        plt.xlabel('Training Steps')
        plt.ylabel('Cumulative Loss')

    # Iterate over the batch
    for i, data in enumerate(batch):
        # Evaluate approximator over interval
        for j, x in enumerate(gtX):
            evalY[j] = approx.evaluate(x)
        # Calculation of the cumulative loss
        cumLoss[i] = cumLoss[i-1] + (approx.evaluate(data[0]) - data[1])**2
        # Perform the learning step
        learner.learn(data[0], data[1])
        # Set time step for target function evaluation (needed for shift and drift)
        funcParam["t"] = i
        # Evaluate target function
        if gTLoss:
            gtY = targetFunc(gtX, noise=0, param=funcParam)

        if plot:
            # Calculation of ground truth loss
            if gTLoss:
                gtLoss[i] = calculateGTloss(targetFunc, approx, funcParam)
                gtPts.set_data(range(i), gtLoss[:i])
            approxPts.set_data(gtX, evalY)
            tfPts.set_data(gtX, gtY)
            ptPts.set_offsets([data[0], data[1]])
            cumLossPts.set_data(range(i), cumLoss[:i])
            plt.pause(0.01)
    if plot:
        plt.show(block=True)
    return cumLoss[-1]


if __name__ == "__main__":
    np.random.seed(12345)
    numData = 1000
    # Target function parameters
    #        len: Length of simulation
    #          t: Current simulation step
    # driftStart: Time step at which the drift starts
    #   driftEnd: Time step at which the drift ends
    funcParam = {"len":numData, "t":0, "driftStart":numData/4, "driftEnd":numData*3/4}
    learnParam = {"l":1.0, "pInit":100000}
    # Uncomment one of the following lines to test your OLLib implementation
#    learnAndPlot(generateSine, numData, GLT, 20, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam)
#    learnAndPlot(generateSine, numData, Polynomial, 20, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam)
#    learnAndPlot(generateSine, numData, GLT, 3, RLSLearner, funcParam, noise=0.1, sort=False, learnParam=learnParam)
    #learnAndPlot(generateSine, numData, Polynomial, 7, RLSLearner, funcParam, noise=0, sort=False, learnParam=learnParam)

    constGLT = []
    constPOLY = []
    linGLT = []
    linPOLY = []
    sinGLT = []
    sinPOLY = []
    stepGLT = []
    stepPOLY = []
    

    for i in range(10):
        print("Iteration ", i, " Const", end="", flush=True)
        constGLT.append(learnAndPlot(generateConstant, numData, GLT, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        constPOLY.append(learnAndPlot(generateConstant, numData, Polynomial, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        print(" Linear", end="", flush=True)
        linGLT.append(learnAndPlot(generateLinear, numData, GLT, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        linPOLY.append(learnAndPlot(generateLinear, numData, Polynomial, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        print(" Sine", end="", flush=True)
        sinGLT.append(learnAndPlot(generateSine, numData, GLT, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        sinPOLY.append(learnAndPlot(generateSine, numData, Polynomial, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        print(" Step")
        stepGLT.append(learnAndPlot(generateStep, numData, GLT, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))
        stepPOLY.append(learnAndPlot(generateStep, numData, Polynomial, 3, PALearner, funcParam, noise=0, sort=False, learnParam=learnParam))

    print(np.mean(constGLT), constGLT)
    print(np.mean(constPOLY), constPOLY)

    print(np.mean(linGLT), linGLT)
    print(np.mean(linPOLY), linPOLY)

    print(np.mean(sinGLT), sinGLT)
    print(np.mean(sinPOLY), sinPOLY)

    print(np.mean(stepGLT), stepGLT)
    print(np.mean(stepPOLY), stepPOLY)
