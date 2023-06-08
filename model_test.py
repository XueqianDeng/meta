import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle

up = pd.read_csv("data/up3.csv")
down = pd.read_csv("data/down3.csv")
left = pd.read_csv("data/left3.csv")
right = pd.read_csv("data/right3.csv")
still = pd.read_csv("data/still3.csv")

up['label'] = ['up'] * up.shape[0]
down['label'] = ['down'] * down.shape[0]
left['label'] = ['left'] * left.shape[0]
right['label'] = ['right'] * right.shape[0]
still['label'] = ['still'] * still.shape[0]

df_test = pd.concat([up, down, left, right, still], ignore_index=True)

#################################################################################
# Compare distributions of training and test data
#################################################################################
# up = pd.read_csv("up.csv")
# down = pd.read_csv("down.csv")
# left = pd.read_csv("left.csv")
# right = pd.read_csv("right.csv")
# still = pd.read_csv("still.csv")
#
# up['label'] = ['up'] * up.shape[0]
# down['label'] = ['down'] * down.shape[0]
# left['label'] = ['left'] * left.shape[0]
# right['label'] = ['right'] * right.shape[0]
# still['label'] = ['still'] * still.shape[0]
#
# df = pd.concat([up, down, left, right, still], ignore_index=True)
#
# fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
# up1 = df.loc[df['label'] == 'up']
# up2 = df_test.loc[df_test['label'] == 'up']
# for i, axis in enumerate(ax.flat):
#     axis.hist(up1.iloc[:, i], bins=20, range=(0, 0.0004), alpha=0.5)
#     axis.hist(up2.iloc[:, i], bins=20, range=(0, 0.0004), alpha=0.5)
# plt.show()


#################################################################################
# Test model with new data
#################################################################################
with open('data/model_MAV.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = df_test.iloc[:, 0:16]
y_test = df_test['label']

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
confuse = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")

labels = ['up', 'down', 'left', 'right', 'still']
disp = ConfusionMatrixDisplay(confusion_matrix=confuse, display_labels=labels)
disp.plot()
plt.show()
