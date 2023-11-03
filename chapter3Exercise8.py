import statsmodels.api as sm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)


Auto = load_data("Auto")
print(Auto.columns)

X = sm.add_constant(Auto['horsepower'])
y = Auto['mpg']
model = sm.OLS(y, X)
results = model.fit()

print(summarize(results))