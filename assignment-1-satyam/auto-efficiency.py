import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read automotive dataset
cols = "mpg,cylinders,displacement,horsepower,weight,acceleration,model year,origin,car name".split(',')
df = pd.read_csv("auto-mpg.data", delim_whitespace="\t", names=cols)

# shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# ignoring car-name as it is uncorrelated to mileage
df = df.drop(columns=['car name'])

# dropping samples with missing values
df = df.drop(index=list(df.query("horsepower == '?'")['horsepower'].index))

# split dataframe into target and features
X = df.drop(columns=['mpg'])
y = df['mpg']

# interpret all features as real continuous
X = X.astype(np.float64)
y = y.astype(np.float64)

# split into training data and testing data
split_idx = int(X.shape[0] * 0.7)
X_train, y_train = X.iloc[:split_idx, :], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:, :], y.iloc[split_idx:]

# Set depth hyperparameter
depth = 5

# train the decision tree
tree = DecisionTree(max_depth=depth)
tree.fit(X_train, y_train)
tree.plot()

# predict
y_hat_train = tree.predict(X_train, verbose=False)
y_hat_test = tree.predict(X_test, verbose=False)

# compare to sklearn
from sklearn.tree import DecisionTreeRegressor, plot_tree

rtree = DecisionTreeRegressor(max_depth=depth)
rtree.fit(X_train, y_train)

y_hat_train_sk = rtree.predict(X_train)
y_hat_test_sk = rtree.predict(X_test)

metrics = pd.DataFrame([[rmse(y_hat_train, y_train), rmse(y_hat_train_sk, y_train)],
                        [mae(y_hat_train, y_train), mae(y_hat_train_sk, y_train)],
                        [r2_score(y_hat_train, y_train), r2_score(y_hat_train_sk, y_train)],
                        [rmse(y_hat_test, y_test), rmse(y_hat_test_sk, y_test)],
                        [mae(y_hat_test, y_test), mae(y_hat_test_sk, y_test)],
                        [r2_score(y_hat_test, y_test), r2_score(y_hat_test_sk, y_test)]], 
                       columns=["Ours", "sklearn"], 
                       index=["Training RMSE", "Training MAE", "Training R^2", "Testing RMSE", "Testing MAE", "Testing R^2"])

print("### Train_Test_Split Results ###")
print(metrics, end="\n\n\n")

# Nested cross-validation
from crossval import nestedCV

logs, mean_outer_error, ctrees = nestedCV(X, y, k_outer=3, k_inner=3, max_depth=10, status=True)
from pprint import pprint
print("OF#  depth")
pprint(logs)

print("Mean RMSE on outer folds (ours):", mean_outer_error)

# compare to sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)

d_grid = {"max_depth" : list(range(1, 10))}
sk_regtree = DecisionTreeRegressor()
reg = GridSearchCV(estimator=sk_regtree, param_grid=d_grid, cv=inner_cv, scoring="r2")
nested_score = cross_val_score(reg, X=X, y=y, cv=outer_cv, scoring='neg_root_mean_squared_error')
print("Scikit-learn mean RMSE scores on outer folds: ", list(map(lambda x: -x, nested_score)))
print("Mean RMSE on outer folds (scikit-learn):", -nested_score.mean())
