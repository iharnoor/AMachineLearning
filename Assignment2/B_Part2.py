import sys
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import the csv data as a dataFrame
inputFilePath = sys.argv[1]  # pass file name of csv as argument
dataFrame = pd.read_csv(inputFilePath)
dataFrame = dataFrame.sort_values('Points', ascending=False)


pd.set_option('display.width', 320)  # dimensions
pd.set_option("display.max_columns", 10)


# Split the data into Training and Testing Data (low = testing data) (high = training data) 50-50
X = dataFrame[["Avg Age", "Players from Outside"]]
Y = dataFrame['Points']
X = sm.add_constant(X)


numOfExperiments = 20
n_list = list(range(numOfExperiments))

# 10 for training data and 10 for testing data
# random teams using random.choices.
avgPointsArray = np.zeros(20)
avgPointsCount = np.zeros(20)

for i in range(numOfExperiments):
    # Random Team
    random_number = random.choices(n_list, k=10)
    list_num_unused = []
    for j in range(numOfExperiments):
        if j not in random_number:
            list_num_unused.append(j)

    ageArray = []
    forArray = []
    pointsArray = []
    for k in random_number:
        ageArray.append(X.loc[k][1])
        forArray.append(X.loc[k][2])
        pointsArray.append(Y.loc[k])

    d = {'Avg Age': ageArray, "Players from Outside": forArray, "Points": pointsArray}
    dfM = pd.DataFrame(data=d)
    dfMX = dfM[["Avg Age", "Players from Outside"]]
    dfMY = dfM['Points']
    dfMX = sm.add_constant(dfMX)
    model = sm.OLS(dfMY, dfMX).fit()

    ageArray = []
    forArray = []
    pointsArray = []
    for k in list_num_unused:
        ageArray.append(X.loc[k][1])
        forArray.append(X.loc[k][2])
        pointsArray.append(Y.loc[k])

    d = {'Avg Age': ageArray, "Players from Outside": forArray, "Points": pointsArray}
    dfL = pd.DataFrame(data=d)
    dfLX = dfL[["Avg Age", "Players from Outside"]]
    dfLY = dfL['Points']
    dfLX = sm.add_constant(dfLX)
    dfLPredictedPoints = model.predict(dfLX)

    i = 0
    for k in list_num_unused:
        avgPointsArray[k] += dfLPredictedPoints[i]
        avgPointsCount[k] += 1
        i += 1

for i in range(len(avgPointsCount)):
    avgPointsArray[i] /= avgPointsCount[i]

y_pred = {"Points": avgPointsArray}
y_pred_dataFrame = pd.DataFrame(data=y_pred)

# plot bottom 10 predicted value and actual value
graph = plt.figure()
graph_3D = Axes3D(graph)

graph_3D.scatter(X['Avg Age'], Y, X["Players from Outside"], color='green')
graph_3D.scatter(X['Avg Age'], y_pred_dataFrame, X["Players from Outside"], color='blue')
plt.legend(['Predicted Points', 'Actual Points'])
graph_3D.set_xlabel('Age')
graph_3D.set_ylabel('Points')
graph_3D.set_zlabel('Foreigners')
plt.show()
