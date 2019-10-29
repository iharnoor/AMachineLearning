import math
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import linalg as SPLA

dataFile = 'HW3.fas'
regex = re.compile('(?<=\n)\w+')


def returnDistanceMatrix(matrix, n_samples):
    distanceMatrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            for k in range(n_samples):
                distanceMatrix[i][j] += math.fabs(matrix[i][k][0] - matrix[j][k][0]) + math.fabs(
                    matrix[i][k][1] - matrix[j][k][1])
                distanceMatrix[j][i] = distanceMatrix[i][j]
    return distanceMatrix


with open(dataFile, 'r') as f:
    data = f.read()

dnaList = regex.findall(data)
dnaMapping = {'A': (0, 1), 'C': (1, 0), 'T': (0, -1), 'G': (-1, 0)}
totals = []
n_samples = len(dnaList)
dnaLength = len(dnaList[0])

# convert each sequence to 2d plot
mappedMatrix = []
for sequence in dnaList:
    i = 0
    plotSequence = []
    for char in sequence:
        plotSequence.append(dnaMapping[char])
    mappedMatrix.append(plotSequence)

# calculate distance between each sequence
# this is the only difference between PCA and MultiDimensional Scalinig. in PCA we use Correlation instead

distanceMatrix = returnDistanceMatrix(mappedMatrix, n_samples)

cov = np.cov(distanceMatrix)
print(cov)

# project in 2d, k = 2
dimensions = 2
eigen_value, eigen_vector = SPLA.eigh(cov, eigvals=(n_samples - dimensions, n_samples - 1))

reducedMatrix = distanceMatrix.dot(eigen_vector)

plt.plot(reducedMatrix, 'ro')
plt.show()
