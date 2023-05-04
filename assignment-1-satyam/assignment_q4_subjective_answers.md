# Answer 4

## Theoretical running time
### `fit()` time:
For the case of **discrete input** (binary), the running time can be calculated via the following recurrence,
$T\left(n, m\right) = 2T\left(\dfrac{n}{2}, m-1\right) + cmn$ where $c$ is a constant.
The term $cmn$ because for all $m$ features, we compute the information gain or reduction in variance which takes $\mathcal{O}(n)$ time. This is non-trivial to solve exactly. For a fixed depth $d$ which is small in comparison to $M$ and $N$, we can approximate $T(N, M) = \mathcal{O}(MN)$.

For the case of **real input**, the running time can be calculated via the following recurrence,
$T\left(n, m\right) = 2T\left(\frac{n}{2}, m\right) + cm(n\log(n) + n^2)$ where $c$ is a constant.
For all $m$ features, we sort which takes $n\log(n)$ time and for all possible $n$ splits, we compute the information gain or reduction in variance in $\mathcal{O}(n)$ time depending on target type, which takes a total of $\mathcal{O}(n^2)$ time. For a fixed depth $d$ which is small in comparison to $M$ and $N$, we can approximate $T(N, M) = \mathcal{O}(MN^2)$.

### `predict()` time

In all the cases, the time complexity is always $\mathcal{O}(dN)$. Once a depth $d$ tree is made, we only need to make $d$ queries per datapoint, for a total of $N$ datapoints.

## Experiment

The values of $N$ range from 50 to 550 in steps of 50, $M$ ranges from $10$ to $50$ in steps of $10$. Each run was repeated 5 times and the mean and standard deviation were taken. All time values are in milliseconds (ms).

|Case|Training time|Training std dev|
|----|-------------|----------------|
|Discrete Input Discrete Output|<img src="plots/tt_DIDO.png" width=400 alt="Training time DIDO">|<img src="plots/tt_std_DIDO.png" height=320 alt="stdev for DIDO">|
|Discrete Input Real Output|<img src="plots/tt_DIRO.png" width=400 alt="Training time DIRO">|<img src="plots/tt_std_DIRO.png" height=320 alt="stdev for DIRO">|
|Real Input Discrete Output|<img src="plots/tt_RIDO.png" width=400 alt="Training time RIDO">|<img src="plots/tt_std_RIDO.png" height=320 alt="stdev for DIRO">|
|Real Input Real Output|<img src="plots/tt_RIRO.png" width=400 alt="Training time RIRO">|<img src="plots/tt_std_RIRO.png" height=320 alt="stdev for RIRO">|

|Case|Testing time|Testing std dev|
|----|------------|---------------|
|Discrete Input Discrete Output|<img src="plots/pt_DIDO.png" width=400 alt="Testing time DIDO">|<img src="plots/pt_std_DIDO.png" height=320 alt="predict() stdev for DIDO">|
|Discrete Input Real Output|<img src="plots/pt_DIRO.png" width=400 alt="Testing time DIRO">|<img src="plots/pt_std_DIRO.png" height=320 alt="predict() stdev for DIRO">|
|Real Input Discrete Output|<img src="plots/pt_RIDO.png" width=400 alt="Testing time RIDO">|<img src="plots/pt_std_RIDO.png" height=320 alt="predict() stdev for DIRO">|
|Real Input Real Output|<img src="plots/pt_RIRO.png" width=400 alt="Testing time RIRO">|<img src="plots/pt_std_RIRO.png" height=320 alt="predict() stdev for RIRO">|

