import math
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import linalg as SPLA

dataFile = 'HW3.fas'
regex = re.compile('(?<=\n)\w+')


def returnCovariationMatrix(dnaList, n_samples):
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


with open(dataFile, 'r') as f:
    data = f.read()

dnaList = regex.findall(data)
n_samples = len(dnaList)
dnaLength = len(dnaList[0])

mappedMatrix = returnCovariationMatrix(dnaList, n_samples)

cov = np.cov(mappedMatrix, rowvar=False)

# number of dimensions to be converted into
dimensions = 2
eigen_value, eigen_vector = SPLA.eigh(cov, eigvals=(dnaLength - dimensions, dnaLength - 1))

print(eigen_value)
print(eigen_vector)

reducedMatrix = mappedMatrix.dot(eigen_vector)

plt.plot(reducedMatrix, 'go')
plt.show()
