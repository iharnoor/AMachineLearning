import sys

import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import matplotlib.pyplot as plt

# import the data as DataFrame
inputFilePath = sys.argv[1]  # pass file name of csv as argument
dataFrame = pd.read_csv(inputFilePath)


# two axis representing average value of players and total points for team
x = dataFrame['Avg Val']
y = dataFrame['Points']

x1 = np.linspace(np.min(x), np.max(x))  # range of evenly distributed points from minx to max(x)


# Total attempts: 20
numOfExperiments = 20
n_List = list(range(numOfExperiments)) # list_Size is sam as the number of experiments


# Creating 10 random teams to dp linear regression and then take average
intercept_avg = 0
slope_avg = 0
for i in range(numOfExperiments):
    random_Num = random.choices(n_List, k=10)  # 10 random numbers from 0 to 20
    x_random = x[random_Num]
    y_random = y[random_Num]
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x_random, y_random)
    intercept_avg += gradient
    slope_avg += intercept

# calculate average intercept and slope
intercept_avg /= numOfExperiments
slope_avg /= numOfExperiments
# create y intercept for predicted values
y_pred = []
for x_pred in x:
    y_pred.append(x_pred * intercept_avg + slope_avg)

# Plot the Regression Line
plt.plot(x1, x1 * intercept_avg + slope_avg, '-r', label='Regression Line')
# plot the Predicted Points
plt.plot(x, y_pred, 'ob', label='Prediction')
# plot original points
plt.plot(x, y, 'og', label='Original Points')
plt.legend()
plt.show()
