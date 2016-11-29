# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:27:20 2016

@author: Jonas Schneider
@author: Alexander Mock
@author: Matthias Greshake
"""

from __future__ import division
import numpy as np
import csv
from perceptron import Perceptron
from multiprocessing import Process, Value, Array, Pool
import argparse
import random
import math


def getNextPic(fileName):
    """
    Iterator that yields all training data line-wise
    @param fileName The name of the file that is to be read
    @return Next line of the specified file as picture and class information
    """
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
            if idx == numLines:
                break
            # yield sample-image as 28x28 pic and the associated class
            yield np.reshape(line[1:], [28,28]).astype(int), int(line[0])


def extract(fileName, outFileName, phi):
    """
    Extract the features of data from an input file to another file
    @param fileName The name of the file that is to be read
    @param outFileName The name of the file that is to be written
    @param phi The function to extract the features from the input file
    """
    outFile = open(outFileName, 'w')
    with open(fileName) as f:
        content = csv.reader(f)
        for idx, line in enumerate(content):
            features = phi(np.reshape(line[1:], [28,28]).astype(int))
            for i in range(len(features) + 1):
                if i < len(features):
                    outFile.write(str(features[i]) + ',')
                else:
                    outFile.write(str(int(line[0])) + '\n')


def readFile(fileName):
    data = []
    with open(fileName) as f:
        content = csv.reader(f)
        for idx, line in enumerate(content):
            data.append(np.array([np.asarray(line[0:8]).astype(float), int(line[8])]))
    return data


def transform(rawData):
    """
    Dummy transformation function (phi(x)) that just calculates the sum of all pixels
    @param rawData The input picture
    @return The input vector in the phi-space
    """
    numBlocksX = 2
    right = np.zeros((numBlocksX, numBlocksX))
    left = np.zeros((numBlocksX, numBlocksX))
    gradient_matrix = np.zeros((len(rawData), len(rawData)))
    gradient_list = []

    for i, row in enumerate(rawData):
        last_pixel = 255
        row_high = []

        for j, pixel in enumerate(row):
#            if np.absolute(pixel - lastpixel) > 50:
            if pixel - last_pixel > 10:
                row_high.append(j)
            last_pixel = pixel

        if len(row_high) > 0:
            point1 = np.array([i, row_high[0]])
            point2 = np.array([i, row_high[len(row_high)-1]])
            gradient_list.append(point1)
            if point1[0] != point2[0]:
                gradient_list.append(point2)
            gradient_matrix[i, row_high[0]] = 1
            gradient_matrix[i, row_high[len(row_high)-1]] = 1

    for row in gradient_matrix:
        grad_str = ""
        for pixel in row:
            grad_str += str(int(pixel))
    direction_list = []
    point_reduce = 2

    while len(gradient_list) > 0:
        point = gradient_list[0]
        del gradient_list[0]
        if len(gradient_list) == 0:
            break
        next_point = nearestNeighbor(point, gradient_list)

        if len(gradient_list) % point_reduce == 0:
            direction_list.append(np.array([np.subtract(point, next_point), np.array(next_point)]))

    old_dirpoint = np.zeros(2)
    for i, entry in enumerate(direction_list):
        dirpoint = entry[0]
        if i == 0:
            old_dirpoint = dirpoint
            continue
        block = getBlock(entry[1], rawData, numBlocksX)
        if isLeftDirection(old_dirpoint, dirpoint):
            left[block[0], block[1]] += 1
        else:
            right[block[0], block[1]] += 1
        old_dirpoint = dirpoint

    result_vec = []
    for i in range(numBlocksX):
        for j in range(numBlocksX):
            result_vec.append(right[i, j])
            result_vec.append(left[i, j])

    return result_vec


def nearestNeighbor(pixel, pixel_list):
    best_length = None
    best_pixel = np.array([0, 0])
    for new_pixel in pixel_list:
        length = np.linalg.norm(np.subtract(pixel, new_pixel))
        if length < best_length or best_length is None:
            best_pixel = new_pixel
            best_length = length
    return best_pixel


def getBlock(point, rawData, numBlocksX):
    length = len(rawData)
    block = np.array([int(point[0] * numBlocksX / length), int(point[1] * numBlocksX / length)])
    return block


def isLeftDirection(old_dirpoint, dirpoint):
    direction3D = np.array([old_dirpoint[0], old_dirpoint[1], 0])
    z = np.array([0, 0, 1])
    n = np.cross(direction3D, z)
    n2D = np.array([n[0], n[1]])
    if np.dot(n2D, dirpoint) > 0:
        return True
    else:
        return False


def calculateError(dataset, perceptron):
    error = 0
    cnt = 0
    for data in dataset:
        _, yh = perceptron.classify(data[0])
        if data[1] != yh:
            error += 1
        cnt += 1
    return error/cnt


def calculateIteratorError(fileName, perceptron, phi):
    """
    Calculate the error on a dataset as percentage wrong classified
    @param fileName The file containing the data
    @param perceptron The perceptron that is to be evaluated
    @param phi The transformation function (phi(x))
    @return The error percentage
    """
    iterator = getNextPic(fileName)
    error = 0
    cnt = 0
    for x, y in iterator:
        _, yh = perceptron.classify(phi(x))
#        print y, yh
        if y != yh:
            error += 1
        cnt += 1
    return error/cnt


if __name__ == "__main__":
    np.random.seed(12345)
#    extract("mnist_first_batch.csv", "mnist_features_batch.csv", transform)
#    extract("mnist_first_test.csv", "mnist_features_test.csv", transform)
    p = Perceptron(8)
    trainData = readFile("mnist_features_batch.csv")
    valData = readFile("mnist_features_test.csv")
    p.learnDataset(trainData, valData, calculateError, maxIterations=3)
    fileName = "mnist_first_test.csv"
    print str(calculateIteratorError(fileName, p, transform) * 100) + '%'
