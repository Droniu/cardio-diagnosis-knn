import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# loading data to tables
file = open("heart.dat")
all = np.loadtxt(file, delimiter=" ")
X = np.zeros((len(all), len(all[0]) - 1), dtype=np.uint8)
y = np.zeros((len(all)), dtype=np.uint8)
for i in range(len(all)):
    for j in range(len(all[0]) - 1):
        X[i][j] = all[i][j]
    y[i] = all[i][len(all[0]) - 1]

features_quantity = len(X[0] - 1)

# feature selection
featureSelection = SelectKBest(f_classif)
featureSelection.fit(X, y)

# sorting by relevance
toSort = np.zeros(
    (len(featureSelection.scores_)), dtype=([("key", "<i4"), ("val", "<f8")])
)
for i in range(len(featureSelection.scores_)):
    toSort[i]["key"] = i + 1
    toSort[i]["val"] = featureSelection.scores_[i]
toSort = np.sort(toSort, order="val")[::-1]

feature_order = []
for i in range(features_quantity):
    print("%d. Feature no. %d, Score: %f" % (i + 1, toSort[i][0], toSort[i][1]))
    feature_order.append(toSort[i][0])
