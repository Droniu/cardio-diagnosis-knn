# Jakub Dudek
# Michał Droń
# knn - dataset statlog heart

#%%writefile "select-filter.py"
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# loading data to tables
file = open("heart.dat")
all = np.loadtxt(file, delimiter=" ")
X = all[:, :-1]
y = all[:, -1].astype(int)
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

from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=10)


from numpy import mean, std
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

metric_list = ["manhattan", "euclidean"]
max_features = 6

for top in range(max_features):
    print("\n")
    # select only top features
    X = X[:, : (top + 1)]

    for p in metric_list:
        for k in range(5, 10, 2):
            model = KNeighborsClassifier(n_neighbors=k, metric=p)
            scores = []
            for train_index, test_index in rskf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                scores.append(accuracy_score(y_test, predict))

            mean_score = mean(scores)
            std_score = std(scores)
            print(
                "Accuracy(k=%d, %s metric, top %d features): %.3f (%.3f)"
                % (k, p, top + 1, mean_score, std_score)
            )

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=42),
}
