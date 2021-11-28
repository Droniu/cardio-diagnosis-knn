import numpy as np
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

# Pobranie danych i zapisanie ich jako X i Y tą lepszą metodą
file = open("heart.dat")
all = np.loadtxt(file, delimiter=" ")
X = all[:, :-1]
y = all[:, -1].astype(int)

features_quantity = len(X[0] - 1)
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

# Deklaracja klasyfikatorów
classifiers = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=42),
    "MLP": MLPClassifier(alpha=1, max_iter=1000),
    "RFC": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "ADA": AdaBoostClassifier(),
}


# 5 powtórzeniowa, dwukrotna walidacja krzyżowa
num_of_splits = 5
num_of_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=num_of_splits, n_repeats=num_of_repeats, random_state=42
)
max_features = 6

for top in range(max_features):
    # deklaracja narazie pustej tablicy wyników klasyfikacji
    scores = np.zeros((len(classifiers), num_of_splits * num_of_repeats))
    X = X[:, : (top + 1)]
    # pętla dla każdego "złożenia"
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        # Dla każdego klasyfikatora
        for classifier_id, classifier_name in enumerate(classifiers):
            classifier = clone(classifiers[classifier_name])  # kopia obiektu
            classifier.fit(X[train], y[train])
            prediction = classifier.predict(X[test])
            scores[classifier_id, fold_id] = accuracy_score(y[test], prediction)
            mean = np.mean(scores, axis=1)
            std = np.std(scores, axis=1)

 
    alfa = 0.05
    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))

    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

    headers = ["GNB", "kNN", "CART", "MLP", "RFC", "ADA"]
    names_column = np.array([["GNB"], ["kNN"], ["CART"], ["MLP"], ["RFC"], ["ADA"]])

    # T_statistics zaznaczenie wartości > 0 (czyli który lepszy od którego)
    advantage = np.zeros((len(classifiers), len(classifiers)))
    advantage[t_statistic > 0] = 1

    # p_value sprawdzenie które różnice są statystycznie znaczące
    significance = np.zeros((len(classifiers), len(classifiers)))
    significance[p_value <= alfa] = 1

    # połączenie powyższych tabel
    stat_better = significance * advantage
    stat_better_table = tabulate(
        np.concatenate((names_column, stat_better), axis=1), headers
    )
    print("For " + str(top + 1) + " features")
    print("Statistically significantly better :\n", stat_better_table)
