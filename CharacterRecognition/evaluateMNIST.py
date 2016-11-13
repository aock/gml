# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:27:20 2016

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
import csv
from perceptron import Perceptron
from multiprocessing import Process, Value, Array, Pool

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


def parallelNearestNeighbor(pixel,pixellist,length_arr,i):
    #for i in range(len(pixellist)):
    length_arr = np.linalg.norm(np.subtract(pixel,pixellist))
    return length_arr

"""
Dummy transformation function (phi(x)) that just calculates the sum of all pixels
@param rawData The input picture
@return The input vector in the phi-space
"""
def nearestNeighbor(pixel,pixellist):
    best_length = 50000
    best_pixel = np.array([0,0])
    
    #length_arr = np.zeros(len(pixellist))
    
       
    
    #for i in range(len(pixellist)):
    #    length_arr[i] = np.linalg.norm(np.subtract(pixel,pixellist[i]))


    #index = np.argmin(length_arr)
    #index = 0    
    #best_pixel = pixellist[index]
        
    
    for pixel_new in pixellist:
        length = np.linalg.norm(np.subtract(pixel,pixel_new)) 
        if length < best_length:
            best_pixel = pixel_new
            best_length = length
    
    return best_pixel
    
def getBlock(point,rawData,numBlocksX):
    length = len(rawData)
    block = np.array([int(point[0]*numBlocksX/length),int(point[1]*numBlocksX/length)])    
    return block
        
def transform(rawData):
    numBlocksX = 2
    right = np.zeros((numBlocksX, numBlocksX))
    left = np.zeros((numBlocksX, numBlocksX))
   
    
    gradientlist = []
    for i,row in enumerate(rawData):
        lastpixel=255
        for j,pixel in enumerate(row):
            if np.absolute(pixel-lastpixel) > 50:
                gradientlist.append(np.array([i,j]))
            lastpixel = pixel
    
    direction_list = []
    
    while len(gradientlist) > 0:
        point = gradientlist[0]
        del gradientlist[0]
        if  len(gradientlist) == 0:
            break
        nearpoint = nearestNeighbor(point, gradientlist)
        
        direction_list.append( np.array([ np.subtract(point,nearpoint) , np.array(nearpoint) ]) )
        
    #print direction_list
    old_dirpoint = np.zeros(2)
    for i,entry in enumerate(direction_list):
        dirpoint = entry[0]
        if i==0:
            old_dirpoint = dirpoint
            continue;
        prod = np.dot(old_dirpoint,dirpoint)
        #print old_dirpoint,dirpoint
        if np.absolute(prod) >= 1:
            block = getBlock(entry[1],rawData,numBlocksX)
            #print block
            if prod < 0:
                left[block[0],block[1]] += 1
            else:
                right[block[0],block[1]] += 1
        old_dirpoint = dirpoint
    
    result_vec = []
    for i in range(numBlocksX):
        for j in range(numBlocksX):
            result_vec.append(right[i,j])
            result_vec.append(left[i,j])
    
        

    norm1 = result_vec / np.linalg.norm(result_vec)    
       
    
    return norm1
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
    #p = Process(target=parallelNearestNeighbor, args=(pixel,pixellist,length_arr))
    #p.start()
    #p.join()    
    
    #for i in range(10):
    p = Perceptron(8, 1)
    print p.learnIteratorDataset(getNextPic, trainData, transform, maxIterations=1)
    print(calculateError(testData, p, phi)*100,'%')
