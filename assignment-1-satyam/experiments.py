import pandas as pd
import numpy as np
np.random.seed(42)
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
import pickle

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

def mean_std_time(tree: DecisionTree, X: pd.DataFrame, y: pd.Series, n_iter: int, status: bool):
    """
    Return average time (and std) taken by fit() and predict() on diven dataset.
    """
    
    fit_times = []
    predict_times = []
    for i in range(n_iter):
        start = time.process_time()
        tree.fit(X, y)
        end = time.process_time()
        delta = (end-start) * 10**3
        fit_times.append(delta)
        if status: print(f"train iter #{i} \ttime={delta} ms")
    fit_mean, fit_std = np.mean(fit_times), np.std(fit_times)
    if status: print(f"mean:{fit_mean} \tstd_dev:{fit_std}")
    for i in range(n_iter):
        start = time.process_time() 
        tree.predict(X)
        end = time.process_time()
        delta = (end-start) * 10**3
        predict_times.append(delta)
        if status: print(f"test iter #{i} \ttime={delta} ms")
    predict_mean, predict_std = np.mean(predict_times), np.std(predict_times)
    if status: print(f"mean:{predict_mean} \tstd_dev:{predict_std}")
    return fit_mean, fit_std, predict_mean, predict_std

def create_fake_data(N: int, M: int, case: str) -> tuple:
    """
    Returns input X and targets y as per case, N samples and M features
    """
    if case == "DIDO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(2, size=N), dtype="category")
    elif case == "RIDO":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(2, size=N), dtype="category")
    elif case == "DIRO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    elif case == "RIRO":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    return X, y

def do_timing_analysis(N_set: list, M_set: list, n_iter: int, depth: int=4, status: bool=True) -> dict:
    """
    Perform running time analysis for the specified case, returns all timing data in a dictionary
    """
    data = {}
    cases = ['DIDO', 'RIDO', 'DIRO', 'RIRO']
    for case in cases:
        data[case] = {}
        if status: print(case)
        for N in N_set:
            data[case][N] = {}
            for M in M_set:
                print(f"N= {N} \t M= {M}")
                data[case][N][M] = {}
                X, y = create_fake_data(N, M, case)
                tree = DecisionTree(max_depth=depth)
                fit_mean, fit_std, predict_mean, predict_std = mean_std_time(tree, X, y, n_iter, status)
                data[case][N][M]['tt_mean'] = fit_mean
                data[case][N][M]['pt_mean'] = predict_mean
                data[case][N][M]['tt_std'] = fit_std
                data[case][N][M]['pt_std'] = predict_std
    return data
                
def get_dataframes(data: dict, N_set: list, M_set: list) -> tuple:
    """
    Get dictionaries of pandas DataFrames from given dictionary of timing data
    tt_dfs, pt_dfs, tt_std_dfs, pt_std_dfs, each is a dictionary with 4 dataframes, one for each case
    """
    M_num, N_num = len(M_set), len(N_set)
    
    df = pd.DataFrame(np.zeros((N_num, M_num)), columns=M_set, index=N_set)
    df.rename_axis("N")
    df.rename_axis("M", axis="columns")
    tt_dfs, pt_dfs, tt_std_dfs, pt_std_dfs = {}, {}, {}, {}

    for case, L1 in data.items():
        tt_dfs[case] = df.copy()
        tt_std_dfs[case] = df.copy()
        pt_dfs[case] = df.copy()
        pt_std_dfs[case] = df.copy()
        for i, (_, L2) in enumerate(L1.items()):
            for j, (_, L3) in enumerate(L2.items()):
                tt_dfs[case].iloc[i].iloc[j] = L3['tt_mean']
                tt_std_dfs[case].iloc[i].iloc[j] = L3['tt_std']
                pt_dfs[case].iloc[i].iloc[j] = L3['pt_mean']
                pt_std_dfs[case].iloc[i].iloc[j] = L3['pt_std']
    return tt_dfs, tt_std_dfs, pt_dfs, pt_std_dfs

def plot_timing_data(tt_dfs: pd.DataFrame, tt_std_dfs: pd.DataFrame, pt_dfs: pd.DataFrame, pt_std_dfs: pd.DataFrame, lims: list, ar: float, save: bool=False):
    cases = ['DIDO', 'RIDO', 'DIRO', 'RIRO']
    for case in cases:
        tt_dfs[case].plot(xlabel="n_samples", ylabel="time (ms)", xticks=tt_dfs[case].index)
        plt.title(f"Training ({case})")
        if save: plt.savefig(f"plots/tt_{case}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        pt_dfs[case].plot(xlabel="n_samples", ylabel="time (ms)", xticks=pt_dfs[case].index)
        plt.title(f"Prediction ({case})")
        if save: plt.savefig(f"plots/pt_{case}.png", dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.imshow(tt_std_dfs[case].to_numpy(), extent=lims, aspect=ar)
        plt.colorbar()
        plt.xticks(tt_std_dfs[case].columns)
        plt.yticks(tt_std_dfs[case].index)
        plt.xlabel("M")
        plt.ylabel("N")
        plt.title(f"train time std_dev ({case})")
        if save: plt.savefig(f"plots/tt_std_{case}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure()
        plt.imshow(pt_std_dfs[case].to_numpy(), extent=lims, aspect=ar)
        plt.colorbar()
        plt.xticks(pt_std_dfs[case].columns)
        plt.yticks(pt_std_dfs[case].index)
        plt.xlabel("M")
        plt.ylabel("N")
        plt.title(f"predict time std_dev ({case})")
        if save: plt.savefig(f"plots/pt_std_{case}.png", dpi=300, bbox_inches='tight')
        plt.show()

N_low = 50
N_high = 550
N_step = 50

M_low = 10
M_high = 50
M_step = 10

N_set = list(range(N_low, N_high + 1, N_step))
M_set = list(range(M_low, M_high + 1, M_step))
lims = [M_low - M_step//2, M_high + M_step//2, N_high + N_step//2, N_low - N_step//2] # extent for imshow
ar = (M_high - M_low) / (N_high - N_low) # aspect ratio for imshow

timing_data = do_timing_analysis(N_set, M_set, n_iter=5)

# save timing data
with open("data/timing_data.pickle", 'wb') as handle:
    pickle.dump(timing_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

tt_dfs, tt_std_dfs, pt_dfs, pt_std_dfs = get_dataframes(timing_data, N_set, M_set)
plot_timing_data(tt_dfs, tt_std_dfs, pt_dfs, pt_std_dfs, lims, ar, save=True)