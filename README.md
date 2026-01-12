## 📂 Repository Structure

```
.
├── README.md
└── The-ANPG-algorithm
    ├── compare_SPG.py
    ├── main_experiment_contaminated.py
    ├── logistic_regression
    │   ├── logistic_regression_algorithms.py
    │   └── main.py

```

---

# 1. Sparse Least Absolute Deviation (LAD) Regression with Outliers

## 1.1 Compared to SPG

### model

We consider the sparse LAD regression problem

$$
\min_{x \in \mathbb{R}^n} 
\frac{1}{m} \Vert Ax - b\Vert_1 + \lambda \, \Phi(x),
$$

where

$$
\Phi(x) = \sum_{i=1}^n \min \( 1, \frac{|x_i|}{\nu}\), \nu>0.
$$

and $x^*$ is a sparse ground truth. In the experiment, 10% of the entries in $b$ are contaminated.

### Algorithms Comparison

- **ANPG (proposed)**  
  Accelerated nested proximal gradient algorithm.

- **GANPG**  
  ANPG with a parameter update rule ensuring sequential convergence of the iterates.

- **SPG** [6] Smoothing proximal gradient algorithm.

### Code
`compare_SPG.py` runs the ANPG/GANPG vs SPG comparison and is ready for direct execution.

---

## 1.2 Comparison of Optimization Models under Contaminated Observations

`main_experiment_contaminated.py` evaluates multiple optimization models in recovering sparse signals under different levels of contamination, where the proportion of corrupted observations is selected from \(\{0\%, 5\%, 10\%, 15\%, 20\%\}\).


### Models and Algorithms:

| Optimization Model    | Solver Algorithm        | Loss Function           | Penalty Function    |
| :-------------------- | :---------------------- | :---------------------- | :------------------ |
| Capped– $l_1$-LAD   | **ANPG (proposed)**     | $\frac{1}{m}\|Ax-b\|_1$ | Capped–$l_1$     |
| LAD-LASSO             | sklearn solver          | $\frac{1}{m}\|Ax-b\|_1$ | $\|x\|_1$           |
| Capped–$l_1$-CS    | HA algorithm [1]        | $\frac{1}{2}\|Ax-b\|_2^2$ | Capped–$\l_1$     |
| $L_{1/2}$-CS          | $\text{IHT}_{1/2}$ algorithm [4] | $\frac{1}{2}\|Ax-b\|_2^2$ | $\|x\|_{1/2}^{1/2}$ |
| $L_0$-CS              | NL0R algorithm [5]      | $\frac{1}{2}\|Ax-b\|_2^2$ | $\|x\|_0$           |
| LASSO                 | sklearn solver          | $\frac{1}{2}\|Ax-b\|_2^2$ | $\|x\|_1$           |
### Note
NL0R requires downloading **CSpack**:  
https://sparseopt.github.io/CS/  
Then import:

```python
from CSpack import CSpack
```

---

# 2. Logistic Regression Experiments

## 2.1 Model

$$
f_{\mathrm{lr}}(x) = \frac{1}{m}\sum_{i=1}^m \left[ \ln(1+\exp(\langle a_i,x\rangle)) - b_i\langle a_i,x\rangle \right],
$$

and the overall objective is

$$
\mathcal{F}(x) = f_{\mathrm{lr}}(x) + \lambda R(x).
$$

## 2.2 Algorithms Implemented

- **APG**: Solves for $R(x) = \Phi(x)$
- **EPDCAe** (Algorithm 2 of [2]): Solves for $R(x) = \Phi(x)$
- **NEPDCA** (Algorithm 1 of [3]): Solves for $R(x) = \Phi(x)$
- **HA algorithm** [1]: Solves for $R(x) = \Phi(x)$
- **IHT$_{1/2}$** [4]: Solves for $R(x) = \|x\|_{1/2}^{1/2}$
- **NL0R** [5]: Solves for $R(x) = \|x\|_0$

## 2.3 Datasets

| Dataset | Training | Test | Features |
|---------|----------|------|----------|
| Wine    | 104      | 26   | 13       |
| Breast  | 455      | 114  | 30       |
| ECG200  | 100      | 100  | 96       |
| Coffee  | 28       | 28   | 286      |
| BirdC   | 20       | 20   | 512      |
| Gisette | 4800     | 1200 | 5000     |
| Leuke   | 30       | 8    | 7129     |
| Duke    | 35       | 9    | 7129     |
| Arcene  | 80       | 20   | 10000    |
| RCV1    | 16193    | 4049 | 47236    |

## 2.4 Running the Logistic Regression Experiments

Download the datasets from the following Google Drive link and save it to your local machine:
https://drive.google.com/file/d/13zEzn52raU8pY4vLZ_GgG3U1Z9RNwFJJ/view?usp=sharing

Set the project root:

```python
PROJECT_ROOT = r"..."
```

Then execute:

```
python logistic_regression/main.py
```

---

# 3. References

[1] **Sun, Z., & Wu, L. (2024).**  
Hybrid algorithms for finding a D-stationary point of a class of structured nonsmooth DC minimization.  
*SIAM Journal on Optimization*, 34(1), 485–506.

[2] **Lu, Z., Zhou, Z., & Sun, Z. (2019).**  
Enhanced proximal DC algorithms with extrapolation for a class of structured nonsmooth DC minimization.  
*Mathematical Programming*, 176(1), 369–401.

[3] **Lu, Z., & Zhou, Z. (2019).**  
Nonmonotone enhanced proximal DC algorithms for a class of structured nonsmooth DC programming.  
*SIAM Journal on Optimization*, 29(4), 2725–2752.

[4] **Xu, Z., Chang, X., Xu, F., & Zhang, H. (2012).**  
\(L_{1/2}\) regularization: A thresholding representation theory and a fast solver.  
*IEEE Transactions on Neural Networks and Learning Systems*, 23(7), 1013–1027.

[5] **Zhou, S., Pan, L., & Xiu, N. (2021).**  
Newton method for \(\ell_0\)-regularized optimization.  
*Numerical Algorithms*, 88(4), 1541–1570.  
Python implementation available at: https://sparseopt.github.io/CS/

[6] **Bian, W., & Chen, X. (2020).**
A smoothing proximal gradient algorithm for nonsmooth convex regression with cardinality penalty.
SIAM Journal on Numerical Analysis, 58(1), 858–883.
