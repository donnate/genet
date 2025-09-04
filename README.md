# genet — Generalized Elastic Net on Graphs

`genet` implements the **Generalized Elastic Net (GEN)** penalty for regression and generalized linear models on graphs (https://arxiv.org/abs/2211.00292).  
It supports solvers based on **Interior-Point (IP)**, **ADMM**, and **Coordinate Gradient Descent (CGD)**, as well as cvxpy backends for GLMs (binomial, Poisson, Gamma, Negative Binomial).

---

## 📦 Installation

### From source (recommended for development)

Clone this repository:

```
git clone https://github.com/your-username/genet.git
cd genet
```


Install in editable mode with dev dependencies:
```
pip install -e ".[dev]"
```

This will install:

- core dependencies (numpy, scipy, scikit-learn, networkx, cvxpy)

- development extras (pytest for testing)

### Using conda

You can also create a clean conda environment:

conda create -n genet python=3.10 -y
conda activate genet
pip install -e ".[dev]



## 🧪 Testing
Run the test suite with:


```
pytest -q
```


## Quickstart

```
import numpy as np
import networkx as nx
from genet.estimators import GenElasticNetEstimator

# synthetic data
n, p = 200, 60
rng = np.random.default_rng(42)
beta_true = np.zeros(p)
beta_true[10:20] = 1.0
X = rng.standard_normal((n, p))
y = X @ beta_true + 0.1 * rng.standard_normal(n)

# graph incidence (path graph)
G = nx.path_graph(p)
D = nx.incidence_matrix(G, oriented=True).T.toarray()

# fit GEN with ADMM solver
gen = GenElasticNetEstimator(l1=0.5, l2=0.2, D=D, family="normal", solver="admm")
gen.fit(X, y)

print("Recovered coefficients shape:", gen.beta.shape)
print("Training score:", gen.score(X, y))
```


## 📖 Documentation

- The genet.estimators module contains:
    - GenElasticNetEstimator (GEN penalty)
    - FusedLassoEstimator, SmoothedLassoEstimator, GTVEstimator
- The genet.solvers module contains:
    - ip_solver,
    - admm_solver,
    - cgd_solver

Example notebooks are provided in the notebooks/ folder.


## 📌 Roadmap

- Add GPU acceleration for ADMM

- Extend support to Tweedie GLMs

- More visualization utilities

## 🤝 Contributing

Pull requests and issues are welcome! Please open a discussion before making major changes.
