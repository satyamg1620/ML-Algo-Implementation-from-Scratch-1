import numpy as np
import pandas as pd
from tree.base import DecisionTree
from metrics import accuracy, rmse

def k_fold_CV(X: pd.DataFrame, y: pd.Series, k: int, max_depth: int, criterion: str = "information_gain") -> tuple:
    """
    Perform k-fold Cross Validation. Returns the average score
    and a list of k trees
    """
    
    df = pd.concat([X, y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset
    X_s = df.drop(columns= [df.columns[-1]])
    y_s = df[df.columns[-1]]
    
    per_fold_accuracy = np.zeros(k)
    ctrees = []
    fold_size = len(X) // k
    for i in range(k):
        X_test = X_s.iloc[i * fold_size : (i + 1) * fold_size]
        y_test = y_s.iloc[i * fold_size : (i + 1) * fold_size]
        
        X_train = pd.concat([X_s.iloc[:i * fold_size], X_s.iloc[(i + 1) * fold_size:]])
        y_train = pd.concat([y_s.iloc[:i * fold_size], y_s.iloc[(i + 1) * fold_size:]])
        
        ctree = DecisionTree(max_depth=max_depth, criterion=criterion)
        ctree.fit(X_train, y_train)
        if y.dtype == "category":
            per_fold_accuracy[i] = accuracy(ctree.predict(X_test), y_test)
        else:
            per_fold_accuracy[i] = rmse(ctree.predict(X_test), y_test)
        ctrees.append(ctree)
    return per_fold_accuracy, ctrees
        
def nestedCV(X: pd.DataFrame, y: pd.Series, k_outer: int, k_inner: int, max_depth: int, criterion: str = "information_gain", status: bool = False):
    """
    Perform nested Cross-Validation to return the mean accuracy and a 
    list of k_outer trees.
    """
    
    if status:
        print(f"{k_outer} x {k_inner} Nested Cross-Validation")
    
    df = pd.concat([X, y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset
    X_s = df.drop(columns= [df.columns[-1]])
    y_s = df[df.columns[-1]]
    
    outer_fold_size = len(df) // k_outer
    
    logs = {}
    trees = []
    mean_outer_accuracy = 0
    for i in range(k_outer):
        # outer fold loop
    
        logs[i] = {}
        
        X_test = X_s.iloc[i * outer_fold_size : (i + 1) * outer_fold_size]
        y_test = y_s.iloc[i * outer_fold_size : (i + 1) * outer_fold_size]
        
        X_train = pd.concat([X_s.iloc[:i * outer_fold_size], X_s.iloc[(i + 1) * outer_fold_size:]])
        y_train = pd.concat([y_s.iloc[:i * outer_fold_size], y_s.iloc[(i + 1) * outer_fold_size:]])
        
        optimum_depth = -1
        if y.dtype == "category":
            max_accuracy = 0
        else:
            max_accuracy = np.inf
    
        for depth in range(max_depth):
            # hyperparameter loop
            
            if status:
                print(f"\rOuter Fold #{i} \t depth={depth}", end="")
                
            logs[i][depth] = {}
            
            # inner fold loop
            per_fold_accuracy, _ = k_fold_CV(X_train, y_train, k_inner, depth, criterion)
            mean_accuracy = per_fold_accuracy.mean()
            
            if y.dtype == "category":
                logs[i][depth]['mean accuracy on IFs'] = mean_accuracy
            
                if logs[i][depth]['mean accuracy on IFs'] > max_accuracy:
                    max_accuracy = logs[i][depth]['mean accuracy on IFs']
                    optimum_depth = depth
            else:
                logs[i][depth]['mean RMSE on IFs'] = mean_accuracy
            
                if logs[i][depth]['mean RMSE on IFs'] < max_accuracy:
                    max_accuracy = logs[i][depth]['mean RMSE on IFs']
                    optimum_depth = depth
        
        # test on outer fold
        tree = DecisionTree(criterion=criterion, max_depth=optimum_depth)
        tree.fit(X_train, y_train)
        y_hat = tree.predict(X_test)
        trees.append(tree)
        logs[i]['optimum_depth'] = optimum_depth
        
        if y.dtype == "category":
            logs[i]['accuracy on OF test'] = round(accuracy(y_hat, y_test), 4)
            mean_outer_accuracy += logs[i]['accuracy on OF test'] / k_outer
        else:
            logs[i]['RMSE on OF test'] = round(rmse(y_hat, y_test), 4)
            mean_outer_accuracy += logs[i]['RMSE on OF test'] / k_outer
    if status: print("\n")
    return logs, mean_outer_accuracy, trees

def ctrees_predict(ctrees: list, X: pd.DataFrame) -> pd.Series:
    """
    Return majority prediction from a list of classification trees.
    """
    
    predictions = []
    for ctree in ctrees:
        predictions.append(ctree.predict(X))
    return pd.concat(predictions, axis=1).mode(axis=1).squeeze()