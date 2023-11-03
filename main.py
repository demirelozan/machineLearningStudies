import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

# Lab section Inputs inside the book
print('fit a model with', 11, 'variables')

a = [3, 4, 5]
print(a)

b = [4, 9, 7]
print(a + b)

x = np.array([3, 4, 5])
y = np.array([4, 9, 7])

print(x + y)

x = np.array([[1, 2], [3, 4]])
x.ndim
x.dtype

np.array([[1, 2], [3.0, 4]]).dtype

x = np.array([1, 2, 3, 4])
np.sum(x)

x = np.array([1, 2, 3, 4, 5, 6])
print('beginning x:\n', x)
x_reshape = x.reshape((2, 3))
print('reshaped x:\n', x_reshape)

print('x before we modify x_reshape:\n', x)
print('x_reshape before we modify x_reshape:\n', x_reshape)
x_reshape[0, 0] = 5
print('x_reshape after we modify its top left element:\n',
      x_reshape)
print('x after we modify top left element of x_reshape:\n', x)

x = np.random.normal(size=50)
print(x)
y = x + np.random.normal(loc=50, scale=1, size=50)
np.corrcoef(x, y)

print(np.random.normal(scale=5, size=2))
print(np.random.normal(scale=5, size=2))

rng = np.random.default_rng(1303)
print(rng.normal(scale=5, size=2))
rng2 = np.random.default_rng(1303)
print(rng2.normal(scale=5, size=2))

# Graphics
fig: object
fig, ax = subplots(figsize=(8, 8))
x = rng.standard_normal(100)
y = rng.standard_normal(100)
ax.plot(x, y);

output = subplots(figsize=(8, 8))
fig = output[0]
ax = output[1]

# Boolean Indexing
A = np.array(np.arange(16)).reshape((4, 4))

keep_rows = np.zeros(A.shape[0], bool)
print(keep_rows)

keep_cols = np.zeros(A.shape[1], bool)
keep_cols[[0, 2, 3]] = True
idx_bool = np.ix_(keep_rows, keep_cols)

# Reading in a Data Set

Auto = pd.read_csv('Auto.data',
                   na_values=['?'],
                   delim_whitespace=True)
Auto['horsepower'].sum()

Auto_new = Auto.dropna()
print(Auto)

Auto = Auto_new  # overwrite the previous value
print(Auto.columns)

idx_80 = Auto['year'] > 80
print(Auto[idx_80])
print(Auto[['mpg', 'horsepower']])

Auto_re = Auto.set_index('name')
print(Auto_re.columns)

# Auto_re.loc[lambda df: (df['displacement'] < 300)
# & (df.index.str.contains('ford')
# | df.index.str.contains('datsun')),
# ['weight', 'origin']
# ]

# For Loops

total: int = 0
for value in [2, 3, 19]:
    for weight in [3, 2, 1]:
        total += value * weight
print('Total is: {0}'.format(total))

total = 0
for value, weight in zip([2, 3, 19],
                         [0.2, 0.3, 0.5]):
    total += weight * value
print('Weighted average is: {0}'.format(total))

rng = np.random.default_rng(1)
A = rng.standard_normal((127, 5))
M = rng.choice([0, np.nan], p=[0.8, 0.2], size=A.shape)
A += M
D = pd.DataFrame(A, columns=['food',
                             'bar',
                             'pickle',
                             'snack',
                             'popcorn'])
D[:3]

for col in D.columns:
    template = 'Column "{0}" has {1:.2%} missing values'
    print(template.format(col,
                          np.isnan(D[col]).mean()))

# Additional Graphical and Numerical Summaries

ax = Auto.plot.scatter('horsepower', 'mpg');
ax.set_title('Horsepower vs. MPG')

fig, axes = subplots(ncols=3, figsize=(15, 5))
Auto.plot.scatter('horsepower', 'mpg', ax=axes[1]);

pd.plotting.scatter_matrix(Auto[['mpg',
                                 'displacement',
                                 'weight']]);

Auto['cylinders'].describe()
Auto['mpg'].describe()
