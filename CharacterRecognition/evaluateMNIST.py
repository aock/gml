# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:27:20 2016

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
import csv
from perceptron import Perceptron

"""
Iterator that yields all training data line-wise
@param fileName The name of the file that is to be read
@return Next line of the specified file as picture and class information
"""
def getNextPic(fileName):
    # Get the total number of lines in the given file
    with open(fileName) as f:
        numLines = sum(1 for _ in f)
    # Iterate over every line (sample)
    with open(fileName) as f:
        # Read comma-seperated-values
        content = csv.reader(f)
        # Iterate over every sample
        for idx,line in enumerate(content):
            # Terminate when eof reached
            if(idx == numLines):
                break
            # yield sample-image as 28x28 pic and the associated class
            yield np.reshape(line[1:], [28,28]).astype(int), int(line[0])

"""
Dummy transformation function (phi(x)) that just calculates the sum of all pixels
@param rawData The input picture
@return The input vector in the phi-space
"""
def transform(rawData, numBlocks=4):
    result = 0
    for i in range(28):
        for j in range(14):
            if rawData[i,j] < 20 and np.absolute(rawData[i,-j]-rawData[i,j]) < 10:
                result += 1
    return [np.sum(rawData), result]
#
#    result = []
#    length = len(rawData)
#    for i in range(0, length, int(length/numBlocks)):
#        for j in range(0, length, int(length/numBlocks)):
#            result.append(np.sum(rawData[i:i+int(length/numBlocks),j:j+int(length/numBlocks)]))
#    return np.asarray(result)
#
#    return [np.sum(rawData)]

"""
Calculate the error on a dataset as percentage wrong classified
@param fileName The file containing the data
@param perceptron The perceptron that is to be evaluated
@param phi The transformation function (phi(x))
@return The error percentage
"""
def calculateError(fileName, perceptron, phi):
    iterator = getNextPic(fileName)
    error = 0
    cnt = 0
    for x, y in iterator:
        _, y_error,yh = perceptron.classify(phi(x),y)
#        print y, y_error
        if y_error < 0:
            error += 1
        cnt += 1
    return error/cnt

def splitData(inputFileName, numTrain, trainData, testData):
    inputFile = open(inputFileName, "r")
    trainDataFile = open(trainData, "w")
    testDataFile = open(testData, "w")
    for i, line in enumerate(inputFile):
        if i < numTrain:
            trainDataFile.write(line)
        else:
            testDataFile.write(line)

if __name__ == "__main__":
    phi = transform
    fileName = "mnist_first_batch.csv"
    trainData = "mnist_first_train.csv"
    testData = "mnist_first_test.csv"
#    splitData(fileName, 30000, trainData, testData)
    for i in range(10):
        p = Perceptron(2, i)
        print p.learnIteratorDataset(getNextPic, trainData, transform, maxIterations=1)
        print(calculateError(testData, p, phi)*100,'%')
