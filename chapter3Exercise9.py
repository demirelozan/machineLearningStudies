import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
from statsmodels.stats.anova import anova_lm

Auto = load_data("Auto")

# Create a scatterplot matrix
from pandas.plotting import scatter_matrix

# a section
scatter_matrix(Auto, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.show()

# b section
corr_matrix = Auto.corr()
print(corr_matrix)

predictors = '+'.join(Auto.columns.difference(['mpg', 'name']))
formula = f'mpg ~ {predictors}'

model = smf.ols(formula=formula, data=Auto).fit()
print(model.summary)

# c section (i)
anova_results = anova_lm(model)
print(anova_results)

# d section
fitted_values = model.fittedvalues
residuals = model.resid
smoothed = sm.nonparametric.lowess(residuals, fitted_values)
top3 = abs(residuals).sort_values(ascending=False)[:3]

plt.figure()
plt.scatter(fitted_values, residuals, edgecolors='k', facecolors='none')
plt.plot(smoothed[:, 0], smoothed[:, 1], color='r')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(True)
plt.show()

# QQ plot
fig = sm.qqplot(residuals, line='45', fit=True)
plt.title('QQ Plot')
plt.show()

# Scale-Location plot
plt.figure()
plt.scatter(fitted_values, np.sqrt(np.abs(residuals)), edgecolors='k', facecolors='none')
plt.xlabel('Fitted values')
plt.ylabel('Square root of |Residuals|')
plt.title('Scale-Location')
plt.grid(True)
plt.show()

# Leverage plot
fig, ax = plt.subplots(figsize=(8, 6))
fig = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
plt.show()


# e section
model_with_interaction = smf.ols(formula='mpg ~ weight + horsepower + weight*horsepower', data=Auto).fit()

print(model_with_interaction.summary())
