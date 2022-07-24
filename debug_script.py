import numpy as np
from scipy import sparse

from sklearn.linear_model import _cd_fast
from sklearn.datasets import make_regression
from sklearn.linear_model.tests.test_sparse_coordinate_descent import make_sparse_data

l1_ratio = 1.
fit_intercept = True
alpha_max_ratio = 0.1
n_samples, n_features = 40, 10

X, y = make_sparse_data(n_samples, n_features)
alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / n_samples

alpha = alpha_max_ratio * alpha_max
l1_reg = alpha * l1_ratio * n_samples
l2_reg = alpha * (1.0 - l1_ratio) * n_samples

X2 = sparse.vstack([X, X[: n_samples // 2]], format="csc")
y2 = np.concatenate([y, y[: n_samples // 2]])
sample_weight = np.ones(n_samples)
sample_weight[: n_samples // 2] = 2


# this doesn't converge
w, gap, *_ = _cd_fast.sparse_enet_coordinate_descent(
    w=np.zeros(n_features, dtype=X.dtype, order="F"),
    alpha=l1_reg,
    beta=l2_reg,
    X_data=X.data,
    X_indices=X.indices,
    X_indptr=X.indptr,
    y=y,
    sample_weight=sample_weight,
    X_mean=np.ones(n_features) if fit_intercept else np.zeros(n_features),
    max_iter=1_000,
    tol=1e-9,
    rng=np.random.RandomState(0),
    random=False,
    positive=False,
)

w2, gap2, *_ = _cd_fast.sparse_enet_coordinate_descent(
    w=np.zeros(n_features, dtype=X.dtype, order="F"),
    alpha=l1_reg,
    beta=l2_reg,
    X_data=X2.data,
    X_indices=X2.indices,
    X_indptr=X2.indptr,
    y=y2,
    sample_weight=None,
    X_mean=np.ones(n_features) if fit_intercept else np.zeros(n_features),
    max_iter=1_000,
    tol=1e-9,
    rng=np.random.RandomState(0),
    random=False,
    positive=False,
)

print(gap, gap2)
print(np.linalg.norm(w - w2, ord=np.inf))
