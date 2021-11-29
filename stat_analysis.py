# Jakub Dudek
# Michał Droń
# knn - dataset statlog heart

#%%writefile "select-filter.py"
import numpy as np
from sklearn import neighbors
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone

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
neighbors = [3, 7, 11]
max_features = 6

for top in range(max_features):
    print("\n")
    # select only top features
    X = X[:, : (top + 6)]
    scores = np.zeros((6, 2 * 5))
    i=0
    for p in metric_list:
        for k in neighbors:
            model = KNeighborsClassifier(n_neighbors=k, metric=p)
            
            for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
                model = clone(model)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                scores[i,fold_id] = accuracy_score(y_test, predict)
            
            mean_score = mean(scores[i])
            std_score = std(scores[i])
            #print('Accuracy(k=%d, %s metric, top %d features): %.3f (%.3f)' % (k, p, top+1, mean_score, std_score))
            i = i+1

            
    from tabulate import tabulate
    from scipy.stats import ttest_rel

    alfa = 0.07
    t_statistic = np.zeros((len(metric_list)*len(neighbors), len(metric_list)*len(neighbors)))
    p_value = np.zeros((len(metric_list)*len(neighbors), len(metric_list)*len(neighbors)))

    for i in range(6):
        for j in range(6):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
    headers = ["manhattan5", "manhattan7", "manhattan9", "euclidean5", "euclidean7", "euclidean9"]
    names_column = np.array([["manhattan5"], ["manhattan7"], ["manhattan9"], ["euclidean5"], ["euclidean7"], ["euclidean9"]])   

    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    #print("For " + str(top + 1) + " features")
    #print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((6, 6))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    #print("For " + str(top + 1) + " features")
    #print("Advantage:\n", advantage_table)
    # p_value sprawdzenie które różnice są statystycznie znaczące
    significance = np.zeros((6,6))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    #print("For " + str(top + 1) + " features")
    #print("Statistical significance (alpha = 0.05):\n", significance_table)
    
    # połączenie powyższych tabel
    stat_better = significance * advantage
    stat_better_table = tabulate(
        np.concatenate((names_column, stat_better), axis=1), headers
    )
    print("For " + str(top + 1) + " features")
    print("Statistically significantly better :\n", stat_better_table)

