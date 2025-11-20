# bo_baseline.py
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from kd_objective import kd_objective

def expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def run_bo_baseline(AF_func, T=3, n_init=3, short=True):
    dim = 4
    def unnormalize(x):
        return {
            "alpha": float(x[0]),
            "beta": float(x[1]) * 0.1,
            "temperature": 2.0 + 18.0 * float(x[2]),
            "learning_rate": 10 ** (-4 + 3.0 * float(x[3])),
            "batch_size": 32,
            "epochs": 3
        }

    X = np.random.rand(n_init, dim)
    y = np.array([kd_objective(unnormalize(x), short=short) for x in X])
    kernel = C(1.0) * RBF(length_scale=np.ones(dim))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    for t in range(T):
        gp.fit(X, y)
        grid = np.random.rand(96, dim)
        mu, sigma = gp.predict(grid, return_std=True)
        vals = AF_func(mu, sigma, np.min(y))
        idx = int(np.argmax(vals))
        x_next = grid[idx]
        y_next = kd_objective(unnormalize(x_next), short=short)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    best_val_acc = 1.0 - float(np.min(y))
    return best_val_acc

if __name__ == "__main__":
    print("Running baseline small test (EI)")
    score = run_bo_baseline(expected_improvement, T=1, n_init=2, short=True)
    print("EI baseline score (val_acc):", score)
