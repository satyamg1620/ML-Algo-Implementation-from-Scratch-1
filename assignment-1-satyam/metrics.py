from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """

    assert y_hat.size == y.size
    return (y_hat == y).sum() / y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    assert y_hat.size == y.size
    tp_fp = (y_hat == cls).sum()
    if tp_fp == 0:
        # all predictions are 'negative', precision in meaningless
        return None
    tp = (y[y_hat == cls] == y_hat[y_hat == cls]).sum()
    return tp / tp_fp


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    assert y_hat.size == y.size
    tp_fn = (y == cls).sum()
    if tp_fn == 0:
        # no 'true' example in ground truth, recall is meaningless
        return None
    tp = (y[y == cls] == y_hat[y == cls]).sum()
    return tp / tp_fn


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size
    return (((y_hat - y) ** 2).mean()) ** 0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    assert y_hat.size == y.size
    return (y_hat - y).abs().mean()


def r2_score(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the R2 regression score
    """

    assert y_hat.size == y.size
    return 1 - ((y_hat - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
