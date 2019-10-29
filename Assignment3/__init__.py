# Part A

import math
import re
import numpy as np

dnaFilePath = 'HW3.fas'
regex = re.compile(
    '(?<=\n)\w+')  # Positive Lookahead regular expression it only takes the characters after the \n which are all characters (AGCT)

# Question 1.
"""Propose how to convert feature vectors of characters into feature vectors of numbers. Justify your approach. Write a script which converts DNA sequences from our dataset into numerical format."""


# For our case we are calculating the distance matrix using
# Our case Manhattan distance seems best as our points are in the form (1,0), (-1,0) etc
def returnDistanceMatrix(matrix, n_samples):
    distanceMatrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            for k in range(n_samples):
                distanceMatrix[i][j] += math.fabs(matrix[i][k][0] - matrix[j][k][0]) + math.fabs(
                    matrix[i][k][1] - matrix[j][k][1])
                distanceMatrix[j][i] = distanceMatrix[i][j]
    return distanceMatrix


# return distance matrix which gives distance between any two points
def dnaToDistanceMatrix():
    with open(dnaFilePath, 'r') as f:
        data = f.read()

    dnaList = regex.findall(data)
    dnaMapping = {'A': (0, 1), 'C': (1, 0), 'T': (0, -1), 'G': (-1, 0)}
    n_samples = len(dnaList)  # total dna rows (120)

    # convert the dna list into a matrix which has each character mapped in the above way  e.g. A becomes (0,1) (2D)
    mappedMatrix = []
    for sequence in dnaList:
        plotSequence = []
        for char in sequence:
            plotSequence.append(dnaMapping[char])
        mappedMatrix.append(plotSequence)

    return returnDistanceMatrix(mappedMatrix, n_samples)


if __name__ == '__main__':
    print(dnaToDistanceMatrix())  # Outputs a distance matrix which is symmetric
