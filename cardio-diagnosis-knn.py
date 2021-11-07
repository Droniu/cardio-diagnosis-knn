#%%writefile "select-filter.py"
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


def complement(arr):
    new_arr = []
    for x in range(len(arr)):
        new_arr.append(abs(arr[x] - len(arr)))
    return np.flip(new_arr)


permutation = complement(feature_order)

idx = np.empty_like(permutation)
idx[permutation] = np.arange(len(permutation))
X = X[:, idx]
print(X)

from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=10)

from numpy import mean, std
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

metric_list = ["manhattan", "euclidean"]
max_features = 6

highest_acc = 0
lowest_acc = 1

# print(X_train[:,:2])
for top in range(max_features):
    print("\n")
    features_train = X[:, : (top + 1)]
    acc_list = []
    for p in metric_list:
        for k in range(5, 10, 2):
            model = KNeighborsClassifier(n_neighbors=k, metric=p)
            scores = cross_val_score(
                model, features_train, y, scoring="accuracy", cv=rkf, n_jobs=-1
            )
            result = mean(scores)
            print(
                "Accuracy(k=%d, %s metric, top %d features): %.3f (%.3f)"
                % (k, p, top + 1, result, std(scores))
            )
            acc_list.append(result)

            if result > highest_acc:
                highest_acc = result
            if result < lowest_acc:
                lowest_acc = result
    print("Mean accuracy for top %d features: %f" % (top + 1, mean(acc_list)))

print("\nMaximum accuracy: %.1f %%" % (highest_acc * 100))
print("Minimum accuracy: %.1f %%" % (lowest_acc * 100))
