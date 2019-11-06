import math
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import linalg as SPLA
from pandas import DataFrame
import random

dataFile = 'HW3.fas'

regex = re.compile('(?<=\n)\w+')  # regex to split


def returnDistanceMatrix(matrix, n_samples):
    distanceMatrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            for k in range(n_samples):
                distanceMatrix[i][j] += math.fabs(matrix[i][k][0] - matrix[j][k][0]) + math.fabs(
                    matrix[i][k][1] - matrix[j][k][1])
                distanceMatrix[j][i] = distanceMatrix[i][j]
    return distanceMatrix


def performKmeans(X, clusters):
    m = X.shape[0]  # number of training examples
    n = X.shape[1]  # number of features. Here n=2
    n_iter = 100

    centroids = np.array([]).reshape(n, 0)

    for i in range(clusters):
        rand = random.randint(0, m - 1)
        centroids = np.c_[centroids, X[rand]]

    predictedList = {}

    euclideanD = np.array([]).reshape(m, 0)  # Euclidean Distance (preferred)

    for k in range(clusters):
        distanceTemp = np.sum((X - centroids[:, k]) ** 2, axis=1)
        euclideanD = np.c_[euclideanD, distanceTemp]

    vectorC = np.argmin(euclideanD, axis=1) + 1

    y_predicted = {}
    for k in range(clusters):
        y_predicted[k + 1] = np.array([]).reshape(2, 0)
    for i in range(m):
        y_predicted[vectorC[i]] = np.c_[y_predicted[vectorC[i]], X[i]]

    for k in range(clusters):
        y_predicted[k + 1] = y_predicted[k + 1].T

    for k in range(clusters):
        centroids[:, k] = np.mean(y_predicted[k + 1], axis=0)

    for i in range(n_iter):
        euclideanD = np.array([]).reshape(m, 0)
        for k in range(clusters):
            distanceTemp = np.sum((X - centroids[:, k]) ** 2, axis=1)
            euclideanD = np.c_[euclideanD, distanceTemp]
        vectorC = np.argmin(euclideanD, axis=1) + 1
        y_predicted = {}
        for k in range(clusters):
            y_predicted[k + 1] = np.array([]).reshape(2, 0)
        for i in range(m):
            y_predicted[vectorC[i]] = np.c_[y_predicted[vectorC[i]], X[i]]
        for k in range(clusters):
            y_predicted[k + 1] = y_predicted[k + 1].T
        for k in range(clusters):
            centroids[:, k] = np.mean(y_predicted[k + 1], axis=0)
        predictedList = y_predicted

    return predictedList, centroids


if __name__ == '__main__':
    with open(dataFile, 'r') as f:
        data = f.read()

    # Solution to question 1. converting into numerical format
    dnaList = regex.findall(data)
    dnaMapping = {'A': (0, 1), 'C': (1, 0), 'T': (0, -1), 'G': (-1, 0)}
    totals = []
    n_samples = len(dnaList)
    dnaLength = len(dnaList[0])


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

    covarianceM = np.cov(distanceMatrix)
    print(covarianceM)

    dimensions = 2
    eigen_value, eigen_vector = SPLA.eigh(covarianceM, eigvals=(n_samples - dimensions, n_samples - 1))

    reducedMatrix = distanceMatrix.dot(eigen_vector)

    dataframe = DataFrame(reducedMatrix)
    X = dataframe.values

    plt.plot(reducedMatrix, 'ro')
    plt.show()

    clusters = 3
    predictedList, centroids = performKmeans(X, clusters)

    plt.scatter(predictedList[1][:, 0], predictedList[1][:, 1], c="red", label="cluster1")
    plt.scatter(predictedList[2][:, 0], predictedList[2][:, 1], c="green", label="cluster2")
    plt.scatter(predictedList[3][:, 0], predictedList[3][:, 1], c="blue", label="cluster3")
    plt.scatter(centroids[0, :], centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.title('Dimensionality Reduction')
    plt.xlabel('DNA Sample')
    plt.ylabel('Multi Dimensional Scaling')
    plt.legend()
    plt.show()
