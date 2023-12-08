import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data

from sklearn.cluster import \
    (KMeans,
     AgglomerativeClustering)
from scipy.cluster.hierarchy import \
    (dendrogram,
     cut_tree)
from ISLP.cluster import compute_linkage

USArrests = get_rdataset('USArrests').data
print(USArrests)

# The columns of the data set contain the four variables
print(USArrests.columns)

#  notice that the variables have vastly diferent means
print(USArrests.mean())

#  examine the variance of the four variables using the var() method.
print(USArrests.var())

# scaling can be done via the StandardScaler(), first fit the scaler, which computes
# the necessary means and standard deviations. Combine these steps using the fit_transform() method
scaler = StandardScaler(with_std=True,
                        with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)

# perform principal components analysis using the PCA()
pcaUS = PCA()

# transform pcaUS can be used to find the PCA scores returned by fit().
pcaUS.fit(USArrests_scaled)

print(pcaUS.mean_)

scores = pcaUS.transform(USArrests_scaled)

# Bi-plot is a common visualization method used with PCA. making a simple bi-plot manually
i, j = 0, 1  # which components
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:, 0], scores[:, 1])
ax.set_xlabel('PC%d' % (i + 1))
ax.set_ylabel('PC%d' % (j + 1))
for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0, pcaUS.components_[i, k], pcaUS.components_[j, k])
    ax.text(pcaUS.components_[i, k],
            pcaUS.components_[j, k],
            USArrests.columns[k])

#  flipping the signs of the second set of scores and loadings.
#  also increasing the length of the arrows to emphasize the loadings.
scale_arrow = s_ = 2
scores[:, 1] *= -1
pcaUS.components_[1] *= -1  # flip the y-axis
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:, 0], scores[:, 1])
ax.set_xlabel('PC%d' % (i + 1))
ax.set_ylabel('PC%d' % (j + 1))
for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0, s_ * pcaUS.components_[i, k], s_ * pcaUS.components_[
        j, k])
    ax.text(s_ * pcaUS.components_[i, k],
            s_ * pcaUS.components_[j, k],
            USArrests.columns[k])

# The standard deviations of the principal component scores
print(scores.std(0, ddof=1))

# The variance of each score can be extracted directly from the pcaUS object
print(pcaUS.explained_variance_)

# The proportion of variance explained by each principal component (PVE):
print(pcaUS.explained_variance_ratio_)

# plot the proportion of variance explained.
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ticks = np.arange(pcaUS.n_components_) + 1
ax = axes[0]
ax.plot(ticks,
        pcaUS.explained_variance_ratio_,
        marker='o')
ax.set_xlabel('Principal Component');
ax.set_ylabel('Proportion of Variance Explained')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)

ax = axes[1]
ax.plot(ticks,
        pcaUS.explained_variance_ratio_.cumsum(),
        marker='o')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Proportion of Variance Explained')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)
print(fig)

# MATRIX COMPLETION

X = USArrests_scaled
U, D, V = np.linalg.svd(X, full_matrices=False)
U.shape, D.shape, V.shape

#  If we multiply each column of U by the corresponding element of D, we recover the PCA scores
# exactly (up to a meaningless sign fip).
(U * D[None, :])[:3]

n_omit = 20
np.random.seed(15)
r_idx = np.random.choice(np.arange(X.shape[0]),
                         n_omit,
                         replace=False)
c_idx = np.random.choice(np.arange(X.shape[1]),
                         n_omit,
                         replace=True)
Xna = X.copy()
Xna[r_idx, c_idx] = np.nan


# t takes in a matrix, and returns an approximation to the matrix using the svd() function
def low_rank(X, M=1):
    U, D, V = np.linalg.svd(X)
    L = U[:, :M] * D[None, :M]
    return L.dot(V[:M])


Xhat = Xna.copy()
Xbar = np.nanmean(Xhat, axis=0)
Xhat[r_idx, c_idx] = Xbar[c_idx]

# set ourselves up to measure the progress of our iterations
thresh = 1e-7
rel_err = 1
count = 0
ismiss = np.isnan(Xna)
mssold = np.mean(Xhat[~ismiss] ** 2)
mss0 = np.mean(Xna[~ismiss] ** 2)

while rel_err > thresh:
    count += 1
# Step 2(a)
Xapp = low_rank(Xhat, M=1)
# Step 2(b)
Xhat[ismiss] = Xapp[ismiss]
# Step 2(c)
mss = np.mean(((Xna - Xapp)[~ismiss]) ** 2)
rel_err = (mssold - mss) / mss0
mssold = mss
print("Iteration: {0}, MSS:{1:.3f}, Rel.Err {2:.2e}"
      .format(count, mss, rel_err))
# compute the correlation between the 20 imputed values and the actual values
print(np.corrcoef(Xapp[ismiss], X[ismiss])[0, 1])

# CLUSTERING

# K-Means Clustering
# begin with a simple simulated example in which there truly are two clusters
# in the data: the frst 25 observations have a mean shift relative to the next 25 observations
np.random.seed(0);
X = np.random.standard_normal((50, 2));
X[:25, 0] += 3;
X[:25, 1] -= 4;

kmeans = KMeans(n_clusters=2,
                random_state=2,
                n_init=20).fit(X)
print(kmeans.labels_)

# plot the data, with each observation colored according to its cluster assignment
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=2");

# K-means clustering on this example with K = 3.
kmeans = KMeans(n_clusters=3,
                random_state=3,
                n_init=20).fit(X)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=3");

# the KMeans() function will report only the
# best results. Here we compare using n_init=1 to n_init=20.
kmeans1 = KMeans(n_clusters=3,
                 random_state=3,
                 n_init=1).fit(X)
kmeans20 = KMeans(n_clusters=3,
                  random_state=3,
                  n_init=20).fit(X);
kmeans1.inertia_, kmeans20.inertia_

# Hierarchical CLustering

# use the data from the previous lab to plot the hierarchical clustering
# dendrogram using complete, single, and average linkage clustering with Euclidean distance
HClust = AgglomerativeClustering
hc_comp = HClust(distance_threshold=0,
                 n_clusters=None,
                 linkage='complete')
hc_comp.fit(X)

#  perform hierarchical clustering with average or single linkage instead:
hc_avg = HClust(distance_threshold=0,
                n_clusters=None,
                linkage='average');
hc_avg.fit(X)
hc_sing = HClust(distance_threshold=0,
                 n_clusters=None,
                 linkage='single');
hc_sing.fit(X);

# To use a precomputed distance matrix, we provide an additional argument metric="precomputed"
D = np.zeros((X.shape[0], X.shape[0]));
for i in range(X.shape[0]):
    x_ = np.multiply.outer(np.ones(X.shape[0]), X[i])
    D[i] = np.sqrt(np.sum((X - x_) ** 2, 1));
hc_sing_pre = HClust(distance_threshold=0,
                     n_clusters=None,
                     metric='precomputed',
                     linkage='single')
hc_sing_pre.fit(D)

# e store these values in a dictionary cargs and pass this as
# keyword arguments using the notation **cargs.
cargs = {'color_threshold': -np.inf,
         'above_threshold_color': 'black'}
linkage_comp = compute_linkage(hc_comp)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp,
           ax=ax,
           **cargs);

#  color branches of the tree above and below a cutthreshold diferently.
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp,
           ax=ax,
           color_threshold=4,
           above_threshold_color='black');

#  determine the cluster labels for each observation associated with a
# given cut of the dendrogram
print(cut_tree(linkage_comp, n_clusters=4).T)

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
hc_comp_scale = HClust(distance_threshold=0,
                       n_clusters=None,
                       linkage='complete').fit(X_scale)
linkage_comp_scale = compute_linkage(hc_comp_scale)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp_scale, ax=ax, **cargs)
ax.set_title("Hierarchical Clustering with Scaled Features");

X = np.random.standard_normal((30, 3))
corD = 1 - np.corrcoef(X)
hc_cor = HClust(linkage='complete',
                distance_threshold=0,
                n_clusters=None,
                metric='precomputed')
hc_cor.fit(corD)
linkage_cor = compute_linkage(hc_cor)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_cor, ax=ax, **cargs)
ax.set_title("Complete Linkage with Correlation-Based Dissimilarity");

# NC160 DATA EXAMPLE
NCI60 = load_data('NCI60')
nci_labs = NCI60['labels']
nci_data = NCI60['data']

print(nci_data.shape)

# We begin by examining the cancer types for the cell lines.
print(nci_labs.value_counts())

# PCA on the NC160 Data
scaler = StandardScaler()
nci_scaled = scaler.fit_transform(nci_data)
nci_pca = PCA()
nci_scores = nci_pca.fit_transform(nci_scaled)

# plot the first few principal component score vectors, in order to visualize the data.
# The observations (cell lines) corresponding to a given cancer type will be plotted in the same color,
# to extend the observations within a cancer type are similar to each other.
cancer_types = list(np.unique(nci_labs))
nci_groups = np.array([cancer_types.index(lab)
                       for lab in nci_labs.values])
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.scatter(nci_scores[:, 0],
           nci_scores[:, 1],
           c=nci_groups,
           marker='o',
           s=50)
ax.set_xlabel('PC1');
ax.set_ylabel('PC2')
ax = axes[1]
ax.scatter(nci_scores[:, 0],
           nci_scores[:, 2],
           c=nci_groups,
           marker='o',
           s=50)
ax.set_xlabel('PC1');
ax.set_ylabel('PC3');

# plot the percent variance explained by the principal components
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ticks = np.arange(nci_pca.n_components_) + 1

ax.plot(ticks,
        nci_pca.explained_variance_ratio_,
        marker='o')
ax.set_xlabel('Principal Component');
ax.set_ylabel('PVE')
ax = axes[1]
ax.plot(ticks,
        nci_pca.explained_variance_ratio_.cumsum(),
        marker='o');
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative PVE');


# Clustering the Observations of the NCI60 Data
def plot_nci(linkage, ax, cut=-np.inf):
    cargs = {'above_threshold_color': 'black',
             'color_threshold': cut}
    hc = HClust(n_clusters=None,
                distance_threshold=0,
                linkage=linkage.lower()).fit(nci_scaled)
    linkage_ = compute_linkage(hc)
    dendrogram(linkage_,
               ax=ax,
               labels=np.asarray(nci_labs),
               leaf_font_size=10,
               **cargs)
    ax.set_title('%s Linkage' % linkage)
    return hc


# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(15, 30))
ax = axes[0];
hc_comp = plot_nci('Complete', ax)
ax = axes[1];
hc_avg = plot_nci('Average', ax)
ax = axes[2];
hc_sing = plot_nci('Single', ax)

# cut the dendrogram at the height that will yield a particular
# number of clusters, say four
linkage_comp = compute_linkage(hc_comp)
comp_cut = cut_tree(linkage_comp, n_clusters=4).reshape(-1)
pd.crosstab(nci_labs['label'],
            pd.Series(comp_cut.reshape(-1), name='Complete'))

# plot a cut on the dendrogram that produces these four clusters
fig, ax = plt.subplots(figsize=(10, 10))
plot_nci('Complete', ax, cut=140)
ax.axhline(140, c='r', linewidth=4);

# perform K-means clustering with K = 4
nci_kmeans = KMeans(n_clusters=4,
                    random_state=0,
                    n_init=20).fit(nci_scaled)
pd.crosstab(pd.Series(comp_cut, name='HClust'),
            pd.Series(nci_kmeans.labels_, name='K-means'))

# perform hierarchical clustering on the frst few principal
# component score vectors, regarding these frst few components as a less
# noisy version of the data.
hc_pca = HClust(n_clusters=None,
                distance_threshold=0,
                linkage='complete'
                ).fit(nci_scores[:, :5])
linkage_pca = compute_linkage(hc_pca)
fig, ax = plt.subplots(figsize=(8, 8))
dendrogram(linkage_pca,
           labels=np.asarray(nci_labs),
           leaf_font_size=10,
           ax=ax,
           **cargs)
ax.set_title("Hier. Clust. on First Five Score Vectors")
pca_labels = pd.Series(cut_tree(linkage_pca,
                                n_clusters=4).reshape(-1),
                       name='Complete-PCA')
pd.crosstab(nci_labs['label'], pca_labels)
