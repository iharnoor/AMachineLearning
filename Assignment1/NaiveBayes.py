# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('amazondata.csv')
X = dataset.iloc[:, [0, 1]].values
# y = dataset.iloc[:, 4].values

print(X)
