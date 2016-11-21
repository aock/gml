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


def extract(fileName, phi):
    """
    Extract the features of data from an input file
    @param fileName The name of the file that is to be read
    @param phi The function to extract the features from the input file
    @return An array with the extracted features and the desired output
    """
    data = []
    with open(fileName) as f:
        content = csv.reader(f)
        for idx, line in enumerate(content):
            data.append(np.array([phi(np.reshape(line[1:], [28,28]).astype(int)), int(line[0])]))
            print line[0]
    return data

def printIntMatrix(m):
    for row in m:
        grad_str = ""
        for pixel in row:
            grad_str += str(int(pixel))
        print grad_str
    
    

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

    #Contour

    for i, row in enumerate(rawData):
        last_pixel = 0
        row_high = []
        
        

        for j, pixel in enumerate(row):
            if np.absolute(pixel - last_pixel) > 50:
 #           if pixel - last_pixel > 10:
                row_high.append(j)
            last_pixel = pixel

        if len(row_high) > 0:
            point1 = np.array([i, row_high[0]])
            point2 = np.array([i, row_high[len(row_high)-1]])
            gradient_list.append(point1)
            if point1[0] != point2[0] or point1[1] != point2[1]:
                gradient_list.append(point2)
            gradient_matrix[i, row_high[0]] = 1
            gradient_matrix[i, row_high[len(row_high)-1]] = 1
            

    point_reduce = 2
    direction_list = []
    counter = 1
    #gradient
    print len(gradient_list) 
    
    index=0
    point = gradient_list[index]
    #brect (min(x,y) , max(x,y))    
    brect = np.zeros((2,2))
    brect[0][0] = 29
    brect[0][1] = 29
    
    while len(gradient_list) > 0:
        
        gradient_matrix[point[0],point[1]] = counter
        counter += 1
        #brect
        if point[0] < brect[0][0]:
            brect[0][0] = point[0]
        if point[1] < brect[0][1]:
            brect[0][1] = point[1]
        if point[0] > brect[1][0]:
            brect[1][0] = point[0]
        if point[1] > brect[1][1]:
            brect[1][1] = point[1]

        del gradient_list[index]

        if len(gradient_list) == 0:
            break        
        
        next_index = nearestNeighbor(point, gradient_list)
        next_point = gradient_list[next_index]

        #if len(gradient_list) % point_reduce == 0:
        direction_list.append(np.array([np.subtract(point, next_point), np.array(next_point)]))
        point = next_point
        index = next_index
            
    gradient_matrix[brect[0][0]][brect[0][1]] = 777
    gradient_matrix[brect[1][0]][brect[1][1]] = 777

    printIntMatrix(gradient_matrix)
    #left right 

    old_dirpoint = np.zeros(2)
    for i, entry in enumerate(direction_list):
        dirpoint = entry[0]
        if i == 0:
            old_dirpoint = dirpoint
            continue
        #block = getBlock(entry[1], rawData, numBlocksX)
        block = getBlock2(entry[1], brect, numBlocksX)
        print block
        
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
    best_index = -1
    for index,new_pixel in enumerate(pixel_list):
        length = np.linalg.norm(np.subtract(pixel, new_pixel))
        if length < best_length or best_length is None:
            best_pixel = new_pixel.copy()
            best_length = length
            best_index = index
    #print pixel
    #print best_pixel
    #print best_length
    
    #print ""
    return best_index


def getBlock(point, rawData, numBlocksX):
    length = len(rawData)
    block = np.array([int(point[0] * numBlocksX / length), int(point[1] * numBlocksX / length)])
    return block
    
def getBlock2(point, brect, numBlocksX):
    lengthX = np.absolute(brect[0][0] - brect[1][0])+1
    lengthY = np.absolute(brect[0][1] - brect[1][1])+1
    
    block = np.array([int((point[0]-brect[0][0]) * numBlocksX / lengthX), int((point[1]-brect[0][1]) * numBlocksX / lengthY)])
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
    for data in dataset:
        _, yh = perceptron.classify(data[0])
        if data[1] != yh:
            error += 1
    return error


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
    p = Perceptron(8)
    fileName = "mnist_first_batch.csv"
    data = extract(fileName, transform)
    p.learnDataset(data, calculateError, maxIterations=1)
    fileName = "mnist_first_test.csv"
    print str(calculateIteratorError(fileName, p, transform) * 100) + '%'
