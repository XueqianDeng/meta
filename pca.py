import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt("data/data.csv", delimiter=',').T
data_abs = np.abs(data)
Nsamples = data.shape[0]
MAV = np.zeros([data.shape[0]-500, data.shape[1]])

idx = 0
for i in range(500, Nsamples):
    MAV[idx, :] = np.mean(data_abs[i-500:i, :], axis=0)
    idx += 1

pca = PCA()
result = pca.fit(MAV)
var = pca.explained_variance_ratio_

pc = pca.components_[0:2, :]
np.savetxt("data/pc.csv", pc, delimiter=",")

plt.subplot(1, 2, 1)
plt.plot(np.arange(1, 17), var)
plt.ylabel('Proportion of explained variance')
plt.xlabel('Principal component number')

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, 17), np.cumsum(var))
plt.ylabel('Proportion of explained variance')
plt.xlabel('Principal component number')
plt.show()

