import numpy as np
from  sklearn.feature_selection  import  SelectKBest, f_classif

# loading data to tables
file = open("heart.dat")
all = np.loadtxt(file, delimiter=" ")
X = np.zeros((len(all),len(all[0])-1),dtype=np.uint8)
Y = np.zeros((len(all)),dtype=np.uint8)
for i in range(len(all)):
    for j in range (len(all[0])-1):
        X[i][j] = all[i][j]
    Y[i] = all[i][len(all[0])-1]
    
# feature selection
featureSelection = SelectKBest(f_classif)
featureSelection.fit(X, Y)
ranks = featureSelection.get_support(True)
for i in ranks:
    print("Feature no.",i+1,"Value ",featureSelection.scores_[i])
