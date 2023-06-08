import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA


df = pd.read_csv("data/point_training.csv")

MAV = df.iloc[:, 0:16]

pca = PCA()
pca.fit(MAV)
print(pca.components_.shape)
# print(pca.explained_variance_ratio_)


# model = NMF(n_components=4, init='random', random_state=0, max_iter=1000)
# W = model.fit_transform(MAV)
# H = model.components_

