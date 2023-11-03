import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)

from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

A = np.array([3, 5, 11])
print(dir(A))

Boston = load_data("Boston")
print(Boston.columns)

X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})
print(X[:4])

y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()

summarize(results)

design = MS(['lstat'])
design = design.fit(Boston)
X = design.transform(Boston)
print(X[:4])

results.summary()

new_df = pd.DataFrame({'lstat': [5, 10, 15]})
newX = design.transform(new_df)
print(newX)

new_predictions = results.get_prediction(newX);
print(new_predictions.predicted_mean)

print(new_predictions.conf_int(alpha=0.05))


def abline(ax, b, m):
    """Add a line with slope m and intercept b to ax"""
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim)

    ax = Boston.plot.scatter('lstat', 'medv')
    abline(ax,
           results.params[0],
           results.params[1],
           'r--',
           linewidth=3)

    ax = subplots(figsize=(8, 8))[1]
    ax.scatter(results.fittedvalues, results.resid)
    ax.set_xlabel('Fitted value')
    ax.set_ylabel('Residual')
    ax.axhline(0, c='k', ls='--');

    infl = results.get_influence()
    ax = subplots(figsize=(8, 8))[1]
    ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
    ax.set_xlabel('Index')
    ax.set_ylabel('Leverage')
    np.argmax(infl.hat_matrix_diag)
