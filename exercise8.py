import pandas as pd
import matplotlib.pyplot as plt

# A section
college = pd.read_csv('College.csv')

print(college.head())

# B section, Getting rid of the Unnamed: 0 column
college2 = pd.read_csv('College.csv', index_col=0)
college3 = college.rename({'Unnamed: 0': 'College'},
                          axis=1)
college3 = college3.set_index('College')
print(college3)
college = college3

# C section
summary = college.describe(include='all')
print(summary)

# D section, selecting the desired columns
columns_to_plot = ['Top10perc', 'Apps', 'Enroll']
subset = college[columns_to_plot]

# Produce the scatterplot matrix
pd.plotting.scatter_matrix(subset, figsize=(10, 10), diagonal='hist', grid=True, alpha=0.5)
# Show the plot
plt.show()

# E section, producing side by side boxplots
college.boxplot(column='Outstate', by='Private', grid=True, figsize=(8, 6))

plt.title('Outstate Tuition by Private/Public College')
plt.suptitle('')  # This removes the default title set by the boxplot method
plt.xlabel('Private')
plt.ylabel('Outstate Tuition')
plt.show()

# F section
college['Elite'] = pd.cut(college['Top10perc'], bins=[0, 50, 100], labels=['No', 'Yes'])
print(college['Elite'].value_counts())

college.boxplot(column='Outstate', by='Elite', grid=True, figsize=(8, 6))
plt.title('Outstate Tuition by Elite Status')
plt.suptitle('')  # removing the default title set by the boxplot method
plt.xlabel('Elite')
plt.ylabel('Outstate Tuition')
plt.show()

# G section
# Setting up the figure and axes for 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Histograms of Selected Variables with Varying Bin Counts')

vars_and_bins = [('Apps', 30), ('Outstate', 50), ('S.F.Ratio', 20), ('Grad.Rate', 40)]

for ax, (var, bin_count) in zip(axes.ravel(), vars_and_bins):
    college[var].plot.hist(bins=bin_count, ax=ax, alpha=0.7, edgecolor='black')
    ax.set_title(f'Histogram of {var} (bins={bin_count})')
    ax.set_xlabel(var)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
