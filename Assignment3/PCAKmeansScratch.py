# PCA Question 2

import re

import matplotlib.pyplot as plt
import numpy as np
import random
from pandas import DataFrame
from scipy import linalg as SPLA

dataFile = 'HW3.fas'

regex = re.compile('(?<=\n)\w+') #regex to split


def returnMappedMatrix(dnaList, n_samples):
    mappedMatrix = np.zeros((n_samples, dnaLength))
    dnaMapping = {'A': 1, 'C': 2, 'T': 3, 'G': 4}

    for i in range(n_samples):
        for j in range(dnaLength):
            mappedMatrix[i][j] = dnaMapping[dnaList[i][j]]

    avg = np.sum(mappedMatrix, axis=0)
    for i in range(dnaLength):
        avg[i] /= n_samples

    for i in range(n_samples):
        for j in range(dnaLength):
            mappedMatrix[i][j] -= avg[j]

    return mappedMatrix


def performKmeans(X, clusters):
    m = X.shape[0]  # number of training examples
    n = X.shape[1]  # number of features. Here n=2
    n_iter = 100

    centroids = np.array([]).reshape(n, 0)

    for i in range(clusters):
        rand = random.randint(0, m - 1)
        centroids = np.c_[centroids, X[rand]]

    predictedList = {}

    euclideanD = np.array([]).reshape(m, 0)  # Euclidean Distance

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

    dnaList = regex.findall(data)
    n_samples = len(dnaList)
    dnaLength = len(dnaList[0])

    mappedMatrix = returnMappedMatrix(dnaList, n_samples)

    # the main difference between Multi dimensional scaling vs PCA here covariance matrix instead of distance matrix
    covariance_Matrix = np.cov(mappedMatrix, rowvar=False)

    # number of dimensions to be converted into
    dimensions = 2  # number of principal components
    eigen_value, eigen_vector = SPLA.eigh(covariance_Matrix, eigvals=(dnaLength - dimensions, dnaLength - 1))

    print(eigen_value)
    print(eigen_vector)

    reducedMatrix = mappedMatrix.dot(eigen_vector)

    dataFrame = DataFrame(reducedMatrix)
    X = dataFrame.values

    plt.plot(X, 'go')
    plt.show()

    clusters = 3
    predictedList, centroids = performKmeans(X, clusters)

    plt.scatter(predictedList[1][:, 0], predictedList[1][:, 1], c="red", label="cluster1")
    plt.scatter(predictedList[2][:, 0], predictedList[2][:, 1], c="green", label="cluster2")
    plt.scatter(predictedList[3][:, 0], predictedList[3][:, 1], c="blue", label="cluster3")

    plt.scatter(centroids[0, :], centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.title('Dimensionality Reduction')
    plt.xlabel('DNA Sample')
    plt.ylabel('PCA')
    plt.legend()
    plt.show()
