import sys
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# use the data from csv file as DataFrame
inputFilePath = sys.argv[1]  # pass file name of csv as argument
dataFrame = pd.read_csv(inputFilePath)
dataFrame = dataFrame.sort_values('Points', ascending=False)


pd.set_option('display.width', 320)
pd.set_option("display.max_columns", 10)


# Split the data into Training and Testing Data (low = testing data) (high = training data) 50-50
x1 = dataFrame['Avg Age']  # Feature1
x1_low = x1[10:]
x1_high = x1[:10]
x2 = dataFrame["Players from Outside"]
x2_low = x2[10:]
x2_high = x2[:10]
y = dataFrame['Points']  # y -> outputt
y_low = y[10:]
y_high = y[:10]

X = dataFrame[["Avg Age", "Players from Outside"]]
Y = y
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

# Training Data: Make predictions for bottom 10
df_bottom10 = dataFrame.sort_values('Points', ascending=False)[10:]

bottom10_X = df_bottom10[['Avg Age', "Players from Outside"]]

bottom10_actualPoints = df_bottom10['Points']
bottom10_XConst = sm.add_constant(bottom10_X)

# make the predictions using the model
bottom10_predictedPoints = model.predict(bottom10_XConst)

# Test data (Plot the predicted results)
graph = plt.figure()
graph_3D = Axes3D(graph)

graph_3D.scatter(x1_high, bottom10_predictedPoints, x2_high, color='blue')
graph_3D.scatter(x1_high, bottom10_actualPoints, x2_high, color='red')
plt.legend(['Predicted Points', 'Actual Points'])
graph_3D.set_xlabel('Age')
graph_3D.set_ylabel('Points')
graph_3D.set_zlabel('Foreigners')
plt.show()
