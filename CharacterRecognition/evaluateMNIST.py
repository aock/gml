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


def transform(rawData):
    """
    Dummy transformation function (phi(x)) that just calculates the sum of all pixels
    @param rawData The input picture
    @return The input vector in the phi-space
    """
    numBlocksX = 2
    right = np.zeros((numBlocksX, numBlocksX))
    left = np.zeros((numBlocksX, numBlocksX))
    gradientmatrix = np.zeros((len(rawData),len(rawData)))
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
            gradientmatrix[i,row_high[0]] = 1
            gradientmatrix[i,row_high[len(row_high)-1]] = 1

    #print("")
    for row in gradientmatrix:
        grad_str = ""
        for pixel in row:
            grad_str += str(int(pixel))
        #print grad_str
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
            continue

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
#    return [np.sum(rawData)]


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


def isLeftDirection(old_dirpoint, dirpoint):

    direction3D = np.array([old_dirpoint[0],old_dirpoint[1],0])

    z = np.array([0,0,1])
    n = np.cross(direction3D,z)
    n2D = np.array([n[0],n[1]])
    if np.dot(n2D,dirpoint) > 0:
        return True
    else:
        return False


def calculateError(fileName, perceptron, phi):
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
    p = Perceptron(8)
    phi = transform
    fileName = "mnist_first_test.csv"
#    p.learnIteratorDataset(getNextPic, fileName, transform, maxIterations=5)
    p.w = np.array([[   2.87829491,  -5.73140841, -9.01309232, -22.63909746, -31.76390496,  -2.71886059,  -7.90989188,  -27.518748,   -44.92102132],
                   [-132.89633861,  11.10185595, 11.76006169,  25.68979536,  12.56970579,  16.35620333,  16.07407413,   12.08717335,   6.49764421],
                   [  30.83712132,  -7.24160436, -0.90831706,  -2.54618851,   8.73432733, -10.87333992,  -7.38270485,  -13.38123191,  -7.755803  ],
                   [ -23.1337966,   -0.42699379,  9.14013299,   2.61481286,   9.01140777,  -4.27843441,   3.8003134,     0.39757907,  10.24256801],
                   [ -19.40132521,   1.09636612, 10.27885506,  -5.23647612, -11.55620119,   0.36309126,   2.42993539,    0.16719847,   1.41959684],
                   [  26.1540506,   -5.6844691,  -4.65461672, -12.00153841, -19.30042594,  -8.02443762,  -1.18092445,   -8.89427464,  -1.59691036],
                   [ -71.75016812,  25.09578447, 29.50525741,  22.81737788,  12.61267854,  -5.60194155,  -8.66093639,  -25.76371193, -32.1713599 ],
                   [ -53.41974128, -18.99121792,-19.53731192,  -8.3212861,   -4.2743606,   10.26110727,  12.8570067,    13.57498452,  14.56112293],
                   [ -47.93280942,   4.40653982, -4.15675345,  -2.94340722, -10.34774916,   1.31707566,   6.21421286,  -11.39310362, -14.47241319],
                   [-122.73930845, 9.59932189, -0.84637331, -18.09101257, -36.66537789, 20.20643921, 18.90436399, 17.04384056, 21.1127725 ]])
    print str(calculateError(fileName, p, phi) * 100) + '%'
