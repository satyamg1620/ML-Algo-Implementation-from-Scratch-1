import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

def get_metrics(prediction, truth):
    metrics = pd.Series(data=[accuracy(prediction, truth), 
                                    precision(prediction, truth, 0), 
                                    precision(prediction, truth, 1), 
                                    recall(prediction, truth, 0), 
                                    recall(prediction, truth, 1)], 
                              index="Accuracy,Precision (Class 0),Precision (Class 1),Recall (Class 0),Recall (Class 1)".split(','))
    return metrics

from sklearn.datasets import make_classification

# Make dataset
n_samples = 100
X, y = make_classification(
    n_samples=n_samples,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2,
    class_sep=0.5,
)

# Plot dataset
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

X = pd.DataFrame(X, dtype=np.float64, columns="X[0],X[1]".split(","))
y = pd.Series(y, dtype="category")

# Split dataset
split_idx = int(n_samples * 0.7)
X_train, y_train = X.iloc[:split_idx, :], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:, :], y.iloc[split_idx:]

# Train classification tree
ctree = DecisionTree(criterion="information_gain", max_depth = 2)
ctree.fit(X_train, y_train)
ctree.plot()

# Predict on train set and test set
y_hat_train = ctree.predict(X_train)
y_hat_test = ctree.predict(X_test)

# Show results
print("\n### Training Metrics ###")
print(get_metrics(y_hat_train, y_train).to_string())
print("Confusion matrix:")
print(pd.crosstab(y_hat_train, y_train, colnames=[''], rownames=['']))

print("\n### Testing Metrics ###")
print(get_metrics(y_hat_test, y_test).to_string())
print("Confusion matrix:")
print(pd.crosstab(y_hat_test, y_test, colnames=[''], rownames=['']), end="\n\n")

### Cross-Validation related functions are written in crossval.py ###
from crossval import k_fold_CV, nestedCV, ctrees_predict

# 5-fold cross-validation
print("### 5-fold cross-validation ###\n")
per_fold_accuracy, ctrees = k_fold_CV(X_train, y_train, k=5, max_depth=2)
print("Accuracy on 5-folds: ", list(per_fold_accuracy.round(3)))
print("Mean accuracy: ", per_fold_accuracy.mean())

# 5 (outer), 4 (inner) fold nested cross-validation
print("\n### Nested Cross-Validation ###\n")
logs, mean_outer_accuracy, ctrees = nestedCV(X_train, y_train, k_inner=4, k_outer=5, max_depth=5, status=True)

from pprint import pprint
pprint(logs)
print("Mean accuracy on outer folds:", mean_outer_accuracy)

# Predict using the trees obtained via nestedCV, take majority
y_hat_ncv = ctrees_predict(ctrees, X_test)
print("Final Accuracy on held test set:", accuracy(y_hat_ncv, y_test))