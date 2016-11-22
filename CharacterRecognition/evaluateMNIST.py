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

import sys

# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


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
    counter = 0
    l = 40000
    with open(fileName) as f:
        content = csv.reader(f)
#        printProgress(counter, l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        for idx, line in enumerate(content):
            features = phi(np.reshape(line[1:], [28,28]).astype(int))
#            if counter%500 ==0:
#                printProgress(counter, l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
#            counter += 1
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
            data.append(np.array([np.asarray(line[0:11]).astype(float), int(line[11])]))

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
    
    #brect (min(x,y) , max(x,y))    
    brect = np.zeros((2,2))
    brect[0][0] = 29
    brect[0][1] = 29
    brect_sizeX = 0
    brect_sizeY = 0

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
            
            #brect
            if point1[0] < brect[0][0]:
                brect[0][0] = point1[0]
            if point1[1] < brect[0][1]:
                brect[0][1] = point1[1]
            if point1[0] > brect[1][0]:
                brect[1][0] = point1[0]
            if point1[1] > brect[1][1]:
                brect[1][1] = point1[1]
                
            if point2[0] < brect[0][0]:
                brect[0][0] = point2[0]
            if point2[1] < brect[0][1]:
                brect[0][1] = point2[1]
            if point2[0] > brect[1][0]:
                brect[1][0] = point2[0]
            if point2[1] > brect[1][1]:
                brect[1][1] = point2[1]
            

    brect[0][0]=0
    brect[0][1]=0
    brect[1][0]=27
    brect[1][1]=27

    point_reduce = 2
    direction_list = []
    counter = 1
    brect_sizeX = int(np.absolute(brect[0][0] - brect[1][0]) + 1)
    brect_sizeY = int(np.absolute(brect[0][1] - brect[1][1]) + 1)
    #print "dhu"    
    #print brect_sizeX
    #print brect_sizeY
    x_histogram = np.zeros( brect_sizeX)
    y_histogram = np.zeros(brect_sizeY )
    #gradient
    
    index=0
    point = gradient_list[index]
    
    
    
    while len(gradient_list) > 0:
        
        #x_histogram[point[0] - brect[0][0]] += 1  
        y_histogram[point[1] - brect[0][1]] += 1
        
        gradient_matrix[point[0],point[1]] = counter
        counter += 1
        

        del gradient_list[index]

        if len(gradient_list) == 0:
            break        
        
        next_index = nearestNeighbor(point, gradient_list)
        next_point = gradient_list[next_index]

        if len(gradient_list) % point_reduce == 0:
            direction_list.append(np.array([np.subtract(point, next_point), np.array(next_point)]))
            point = next_point
        index = next_index
            
    gradient_matrix[brect[0][0]][brect[0][1]] = 777
    gradient_matrix[brect[1][0]][brect[1][1]] = 777

    #printIntMatrix(gradient_matrix)
    #left right 
    peek_index1 = np.argmax(y_histogram)
    peek_position1 = peek_index1/brect_sizeY
    y_histogram[peek_index1] = 0
    
    peek_index2 = np.argmax(y_histogram)
    peek_position2 = peek_index2/brect_sizeY
    y_histogram[peek_index2] = 0    
    
    #print y_histogram
    #print peek_position1
    #print peek_position2

    old_dirpoint = np.zeros(2)
    for i, entry in enumerate(direction_list):
        dirpoint = entry[0]
        if i == 0:
            old_dirpoint = dirpoint
            continue
        #block = getBlock(entry[1], rawData, numBlocksX)
        block = getBlock2(entry[1], brect, numBlocksX)
        
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
    
    result_vec.append(peek_position1)
    result_vec.append(peek_position2)
    result_vec.append(brect_sizeX/brect_sizeY) 
    #print result_vec
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
#    extract("mnist_first_val.csv", "mnist_features_val.csv", transform)
    p = Perceptron(11)
    trainData = readFile("mnist_features_batch.csv")
    valData = readFile("mnist_features_val.csv")

    p.learnDataset(trainData, valData, calculateError, maxIterations=2)
    print str(calculateError(valData, p) * 100) + '%'
