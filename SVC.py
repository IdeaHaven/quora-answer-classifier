import pandas as pd
from sklearn import svm

train = pd.read_csv('./train-plucked.csv')

classifiers = train.ix[:, -1]
features = train.ix[:, 1:-1]

clf = svm.SVC(verbose=True)
print clf.fit(features, classifiers)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False, shrinking=True, tol=0.001, verbose=True)
