import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle

up = pd.read_csv("data/up.csv")
down = pd.read_csv("data/down.csv")
left = pd.read_csv("data/left.csv")
right = pd.read_csv("data/right.csv")
still = pd.read_csv("data/still.csv")

up['label'] = ['up'] * up.shape[0]
down['label'] = ['down'] * down.shape[0]
left['label'] = ['left'] * left.shape[0]
right['label'] = ['right'] * right.shape[0]
still['label'] = ['still'] * still.shape[0]

df = pd.concat([up, down, left, right, still], ignore_index=True)

####################################################################
# Visualize data using PCA
####################################################################
# pca = PCA(n_components=2)
# pc = pca.fit_transform(df.iloc[:, 0:16])
#
# df_transform = pd.DataFrame(data=pc, columns=['PC1', 'PC2'])
# df_transform['label'] = df['label']
#
# print(df_transform)
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('2 component PCA', fontsize=20)
#
# targets = ['up', 'down', 'left', 'right']
# colors = ['r', 'g', 'b', 'k']
# for target, color in zip(targets, colors):
#     print(df_transform['label'])
#     indicesToKeep = df_transform['label'] == target
#     ax.scatter(df_transform.loc[indicesToKeep, 'PC1']
#                , df_transform.loc[indicesToKeep, 'PC2']
#                , c=color
#                , s=50)
# ax.legend(targets)
# ax.grid()
# plt.show()


#################################################################
# Fit SVM
#################################################################
fit_model = False

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:16], df['label'], test_size=0.33, random_state=42)

if fit_model:
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    model = svm.fit(X_train, y_train)
    with open('data/model_MAV.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('data/model_MAV.pkl', 'rb') as f:
        model = pickle.load(f)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
max_prob = np.amax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
confuse = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")

labels = ['up', 'down', 'left', 'right', 'still']
disp = ConfusionMatrixDisplay(confusion_matrix=confuse, display_labels=labels)
disp.plot()
plt.show()

plt.figure(2)
plt.hist(max_prob)
plt.show()

