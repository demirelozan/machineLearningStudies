import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize)
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA,
     QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Smarket = load_data('Smarket')
print(Smarket)

print(Smarket.columns)

print(Smarket.corr())

Smarket.plot(y='Volume');

allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm = sm.GLM(y,
             X,
             family=sm.families.Binomial())
results = glm.fit()
summarize(results)

print(results.params)

probs = results.predict()
probs[:10]

labels = np.array(['Down'] * 1250)
labels[probs > 0.5] = "Up"

print(confusion_table(labels, Smarket.Direction))

print((507 + 145) / 1250, np.mean(labels == Smarket.Direction))

train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
print(Smarket_test.shape)

X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
print(probs)

D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

labels = np.array(['Down'] * 252)
labels[probs > 0.5] = 'Up'
confusion_table(labels, L_test)

np.mean(labels == L_test), np.mean(labels != L_test)

model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down'] * 252)
labels[probs > 0.5] = 'Up'
print(confusion_table(labels, L_test))

print((35 + 106) / 252, 106 / (106 + 76))

newdata = pd.DataFrame({'Lag1': [1.2, 1.5],
                        'Lag2': [1.1, -0.8]});
newX = model.transform(newdata)
print(results.predict(newX))

lda = LDA(store_covariance=True)
X_train, X_test = [M.drop(columns=['intercept'])
                   for M in [X_train, X_test]]
lda.fit(X_train, L_train)

print(lda.means_)
print(lda.classes_)
print(lda.priors_)
print(lda.scalings_)

lda_pred = lda.predict(X_test)
print(lda_pred)

print(confusion_table(lda_pred, L_test))
lda_prob = lda.predict_proba(X_test)
np.all(
    np.where(lda_prob[:, 1] >= 0.5, 'Up', 'Down') == lda_pred
)
np.all(
    [lda.classes_[i] for i in np.argmax(lda_prob, 1)] ==
    lda_pred
)

np.sum(lda_prob[:, 0] > 0.9)

qda = QDA(store_covariance=True)
qda.fit(X_train, L_train)
print(qda.means_, qda.priors_)
qda.covariance_[0]

qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)

np.mean(qda_pred == L_test)

NB = GaussianNB()
NB.fit(X_train, L_train)
print(NB.classes_)
print(NB.class_prior_)
print(NB.theta_)
print(NB.var_)
print(X_train[L_train == 'Down'].mean())

X_train[L_train == 'Down'].var(ddof=0)

nb_labels = NB.predict(X_test)
print(confusion_table(nb_labels, L_test))
print(NB.predict_proba(X_test)[:5])

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
print(confusion_table(knn1_pred, L_test))
print((83 + 43) / 252, np.mean(knn1_pred == L_test))

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3_pred = knn3.fit(X_train, L_train).predict(X_test)
np.mean(knn3_pred == L_test)

Caravan = load_data('Caravan')
Purchase = Caravan.Purchase
Purchase.value_counts()

feature_df = Caravan.drop(columns=['Purchase'])

scaler = StandardScaler(with_mean=True,
                        with_std=True,
                        copy=True)

scaler.fit(feature_df)
X_std = scaler.transform(feature_df)

feature_std = pd.DataFrame(
    X_std,
    columns=feature_df.columns);
print(feature_std.std())

(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(feature_std,
                            Purchase,
                            test_size=1000,
                            random_state=0)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
np.mean(y_test != knn1_pred), np.mean(y_test != "No")

print(confusion_table(knn1_pred, y_test))

for K in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    C = confusion_table(knn_pred, y_test)
    templ = ('K={0:d}: # predicted to rent: {1:>2},' +
             ' # who did rent {2:d}, accuracy {3:.1%}')
    pred = C.loc['Yes'].sum()
    did_rent = C.loc['Yes', 'Yes']
    print(templ.format(
        K,
        pred,
        did_rent,
        did_rent / pred))
