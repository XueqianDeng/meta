################################################
# Train a model to predict cursor position from MAV
#
# Training data is generated by pointing_collection.py
################################################

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

converter = 0
con_name = ['vec_1.csv', 'vec_2.csv', 'vec_3.csv', 'vec_4.csv']

df = pd.read_csv('data/' + con_name[converter])
X = df.iloc[501: , :16]  # MAV
Y = df.iloc[501: , 16:18]  # target positions
time = df.iloc[:, -1]  # time

print(X)

X_matrix = X.values

print(X_matrix)

mean_values = np.mean(X_matrix, axis=0)

print("Mean Value", mean_values)
np.savetxt('data/' + con_name[converter], mean_values, delimiter=',')

# print(Y)