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
import argparse
import random
import math

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

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))
  
def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
  
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
    #print(str(point[0]) +","+str(point[1])+": " + str(block[0])+","+str(block[1]))      
    return block
        
def transform(rawData):
    numBlocksX = 2
    right = np.zeros((numBlocksX, numBlocksX))
    left = np.zeros((numBlocksX, numBlocksX))
   
    
    gradientlist = []
    for i,row in enumerate(rawData):
        lastpixel=255
        for j,pixel in enumerate(row):
            #if np.absolute(pixel-lastpixel) > 50:
            if pixel-lastpixel > 50:
                gradientlist.append(np.array([i,j]))
            lastpixel = pixel
    
    direction_list = []
    point_reduce = 2  
    
    while len(gradientlist) > 0:
        point = gradientlist[0]
        del gradientlist[0]
        if  len(gradientlist) == 0:
            break
        nearpoint = nearestNeighbor(point, gradientlist)
        
        if len(gradientlist)%point_reduce == 0:
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
        if np.absolute(prod) >= 0:
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
       
    
    return result_vec

def transform_base(rawData):
    numBlocksX = 2
    
    #result = []
    #length = len(rawData)
    #for i in range(0, length, int(length/numBlocks)):
    #    for j in range(0, length, int(length/numBlocks)):
    #        result.append(np.sum(rawData[i:i+int(length/numBlocks),j:j+int(length/numBlocks)]))
    #return np.asarray(result)

    return [np.sum(rawData)]

def isLeftDirection(old_dirpoint, dirpoint):
    
    direction3D = np.array([old_dirpoint[0],old_dirpoint[1],0])
    
    z = np.array([0,0,1])
    n = np.cross(direction3D,z)
    n2D = np.array([n[0],n[1]])
    if np.dot(n2D,dirpoint) > 0:
        return True
    else:
        return False
    
    return

def transformB(rawData):
    numBlocksX = 2
    right = np.zeros((numBlocksX, numBlocksX))
    left = np.zeros((numBlocksX, numBlocksX))
    
    gradientlist = []
    for i,row in enumerate(rawData):
        lastpixel=255
        
        row_high = []
        for j,pixel in enumerate(row):
            #if np.absolute(pixel-lastpixel) > 50:
            if pixel-lastpixel > 10:
                row_high.append(j)
            lastpixel = pixel
        if len(row_high) > 0 :
            point1 = np.array([i,row_high[0]])
            point2 = np.array([i,row_high[len(row_high)-1]])
            gradientlist.append(point1)
            if point1[0] != point2[0] or point1[0] != point2[0]:
                gradientlist.append(point2)
                
      
    direction_list = []
    point_reduce = 2  
    
    while len(gradientlist) > 0:
        point = gradientlist[0]
        del gradientlist[0]
        if  len(gradientlist) == 0:
            break
        nearpoint = nearestNeighbor(point, gradientlist)
        
        if len(gradientlist)%point_reduce == 0:
            direction_list.append( np.array([ np.subtract(point,nearpoint) , np.array(nearpoint) ]) )
        
    #print direction_list
    old_dirpoint = np.zeros(2)
    for i,entry in enumerate(direction_list):
        dirpoint = entry[0]
        if i==0:
            old_dirpoint = dirpoint
            continue;
        
        #print prod
        block = getBlock(entry[1],rawData,numBlocksX)
        #print block
        #print prod
        if isLeftDirection(old_dirpoint,dirpoint):
            left[block[0],block[1]] += 1
        else:
            right[block[0],block[1]] += 1
        old_dirpoint = dirpoint    
    
    result_vec = []
    #print right
    #print left
    #print ""
    for i in range(numBlocksX):
        for j in range(numBlocksX):
            result_vec.append(right[i,j])
            result_vec.append(left[i,j])
            
    return result_vec


def evaluate(datum, weight_vecs, phi):
    
    y_h = []        
    
    perceptrons = []
    for i in range(len(weight_vecs)):
        perceptrons.append(Perceptron(8,i))
        perceptrons[i].w = weight_vecs[i]
        _,yh = perceptrons[i].classify(phi(x))
        #print y_error        
        y_h.append(yh)
    
    print y_h
    
    return 0
    

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
        if y == perceptron.num:    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", help="Learn Number")
    parser.add_argument("-i", "--iterations", help="Dataset Iterations")  
    parser.add_argument("-e", "--evaluate", help="Evaluate Datum")  
    
    args = parser.parse_args()

    
    phi = transformB
    fileName = "mnist_first_batch.csv"
    trainData = "mnist_first_train.csv"
    testData = "mnist_first_test.csv"
    number = 1    
    iterations = 1
    if args.number:
        number = int(args.number)
    if args.iterations:
        iterations = int(args.iterations)
#    splitData(fileName, 30000, trainData, testData)
    #p = Process(target=parallelNearestNeighbor, args=(pixel,pixellist,length_arr))
    #p.start()
    #p.join()    
    
    #for i in range(10):    
    p = Perceptron(8, number)
    print("Start learning the number: "+str(number)+ ", with "+str(iterations)+ " Iterations")
    print p.learnIteratorDataset(getNextPic, trainData, phi, maxIterations=iterations)
    print(calculateError(testData, p, phi)*100,'%')

    #TEST VECTOR
    #example weigth vec for 2 
    
    # 9.66%
    zero = [-514.17706066,19.42001634,27.84531342,7.12038749,-9.52344336,20.94214496,-4.0804675,10.6551297,24.52467324]
    # 7.79%
    one = [302.64661193,-27.73111528,-63.49517691,0.89106104,-9.92501237,-14.6691409,-44.85823597,-30.07874099,-73.65624532]
    # 9.95%
    two = [-59.42937599,-5.46154605,1.31207586,10.44606889,-2.93922239,-10.47353565,17.89622458,-7.63600549,10.83191407]
    # 14.37%     
    three = [6.87819836,-7.40261509,-8.58351018,4.08096154,5.60154467,-3.39220905,-3.80732736,-5.59883534,-5.08279869]
    # 9.6%    
    four = [-69.09672892,12.5314921,-8.69069143,10.42321623,6.65761534,-19.17412318,-10.42162825,-3.85863103,-9.16652308]
    # 9.65%   
    five = [ 63.62104261,-7.32430134,4.6709081,-26.88837134,-0.35461544,-1.23544655,3.28730578,-2.69538322,5.33241149]
    # 23.32%    
    six = [-138.10197872,15.54776885,17.53386306,-7.38828945,1.85694135,7.29347097,8.90298238,-1.28234472,13.89760125]
    # 11.06% 
    seven = [ 79.24840391,-25.2052897,-25.42384614,-17.80834112,-20.88616073,-3.04920832,-4.15417432,9.456639,20.29626943]
    # 12,56%    
    eight = [-355.1522296,17.368128,3.32093904,3.88785582,27.2373897,27.73570413,15.93431831,-8.87876971,-15.57680764]
    # 10.14
    nine = [-75.22144676,9.32316996,11.42960455,-7.78341425,12.83026267,-13.61039633,-13.54682615,-1.95401298,11.75910704]

    #iterator = getNextPic(testData)
    #index = 0
    #if args.evaluate:
        #index = int(args.evaluate)
    #for x,y in iterator:
        #if index != 0:
            #index -= 1
            
        #else:
            
            #print x,y
            #value = evaluate(x,[zero],transform)#,one,two,three,four,five,six,seven,eight,nine],transform)
            #print value
            #break
        

            
    
