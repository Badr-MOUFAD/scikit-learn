import numpy as np

from sklearn.linear_model import _cd_fast
from sklearn.datasets import make_regression

from sklearn.linear_model import ElasticNet


n_samples, n_features = 30, 50
X, y = make_regression(n_samples, n_features, bias=1)

l1_ratio = 1.
fit_intercept = True
alpha_max_ratio = 0.1


center = float(fit_intercept)
alpha_max = np.linalg.norm(X.T - center * np.mean(X, axis=1) @ (y - center * np.mean(y)),
                           ord=np.inf) / n_samples

alpha = alpha_max_ratio * alpha_max
l1_reg = alpha * l1_ratio * n_samples
l2_reg = alpha * (1.0 - l1_ratio) * n_samples

params = {'tol': 1e-9, 'random': False,
          'positive': False, 'max_iter': 10000,
          'rng': np.random.RandomState(0)}

coef = np.zeros(n_features, dtype=X.dtype, order="F")
X = np.asfortranarray(X, dtype=X.dtype)
y = np.asfortranarray(y, dtype=y.dtype)

w, gap, tol, n_iter = _cd_fast.enet_coordinate_descent(
    coef, l1_reg, l2_reg, X, y, fit_intercept=fit_intercept, **params
)


model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                   fit_intercept=fit_intercept, tol=1e-9)
model.fit(X, y)


print(np.linalg.norm(w - model.coef_, ord=np.inf))
