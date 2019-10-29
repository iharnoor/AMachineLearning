import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# import the CSV data as DataFrame
inputFilePath = sys.argv[1]  # pass file name of csv as argument
dataFrame = pd.read_csv(inputFilePath)


# Sort in descending order according to the points
dataFrame = dataFrame.sort_values('Points', ascending=False)


# print to console for reference
pd.set_option('display.width', 320)
pd.set_option("display.max_columns", 10)

# split data into Training and Test Data/ Higher value data will be training data and low will testing data
x = dataFrame['Avg Val'].values
x_low = x[10:]  # low from index 10, testing data
x_high = x[:10]  # high till index
y = dataFrame['Points'].values
y_low = y[10:]
y_high = y[:10]

# plot high in red low in blue
plt.plot(x_high, y_high, 'or', label='Training Data')  # o for circles and r for red (so red circles
plt.plot(x_low, y_low, 'ob', label='Test Data')  # so blue circles

# linear regression for top 10 high values
slope, y_intercept, r_value, p_value, std_err = stats.linregress(x_high, y_high)
mn = np.min(x)
mx = np.max(x)
x_max = np.linspace(mn, mx)  # range of datapoints, evenly distributed
y_max = slope * x_max + y_intercept

# calculate predicted y (value in csv)
y_pred = []
for x in x_low:
    y_pred.append(x * slope + y_intercept)

print(y_pred)

plt.plot(x_low, y_pred, 'og', label='Prediction')
plt.plot(x_max, y_max, '-r', label='Regression Line')
plt.xlabel('Avg Value')
plt.ylabel('Points')
plt.legend()
plt.show()
