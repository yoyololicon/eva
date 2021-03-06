import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

labels = pd.read_csv('annotations.csv')
label_names = labels.columns.values[1:]
file_names = labels['File Name'].values
labels = labels.values[:, 1:]
num2class = dict(enumerate(label_names))

class_nums = labels.sum(axis=0)
for i in range(len(label_names)):
    print(num2class[i], ":", class_nums[i], "of samples.")

features = pd.read_csv('features.csv', index_col=0, header=[0, 1, 2])
print("Features shape:", features.shape)
assert features.shape[0] == len(file_names) == class_nums.sum()

simple_labels = labels[:, :2] + labels[:, 2:4]
simple_labels = np.column_stack((simple_labels, labels[:, -1]))

y = np.nonzero(labels)[1]
y_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
X = LDA(n_components=2).fit_transform(features, y)

figsize = (8, 6)
plt.figure(figsize=figsize)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
ax = []
for c, i in zip(colors[:y.max() + 1], range(y.max() + 1)):
    idx = np.where(y == i)
    ax.append(plt.scatter(X[idx, 0], X[idx, 1], c=c, alpha=0.5))
plt.legend(ax, [num2class[i] for i in range(y.max() + 1)])
plt.show()

sy = np.nonzero(simple_labels)[1]
sy_weights = dict(enumerate(compute_class_weight('balanced', np.unique(sy), sy)))
sX = LDA(n_components=2).fit_transform(features, sy)

plt.figure(figsize=figsize)
ax = []
for c, i in zip(colors[:sy.max() + 1], range(sy.max() + 1)):
    idx = np.where(sy == i)
    ax.append(plt.scatter(sX[idx, 0], sX[idx, 1], c=c, alpha=0.5))
plt.legend(ax, [num2class[i + 2] for i in range(sy.max() + 1)])
plt.show()

feature_sets = {'mfcc': ['mfcc'],
                'mfcc/contrast': ['mfcc', 'spectral_contrast'],
                'mfcc/centroid': ['mfcc', 'spectral_centroid'],
                'mfcc/bandwidth': ['mfcc', 'spectral_bandwidth'],
                'mfcc/flatness': ['mfcc', 'spectral_flatness'],
                'mfcc/rolloff': ['mfcc', 'spectral_rolloff'],
                'mfcc/bandwidth/zcr': ['mfcc', 'spectral_bandwidth', 'zcr'],
                'mfcc/bandwidth/rmse': ['mfcc', 'spectral_bandwidth', 'rmse'],
                'all_features': list(features.columns.levels[0])}

clf = SVC(kernel='linear')

print("With original labels:")
X_train, X_test, y_train, y_test, sy_train, sy_test = train_test_split(features, y, sy, test_size=0.3)
for fset_name, fset in sorted(feature_sets.items(), key=lambda x: len(x[0])):
    clf.fit(X_train.loc[:, fset], y_train)
    weights = compute_sample_weight('balanced', y_test)
    acc = clf.score(X_test.loc[:, fset], y_test, sample_weight=weights)
    print(fset_name + ":", acc)

print("With simplified labels:")
for fset_name, fset in sorted(feature_sets.items(), key=lambda x: len(x[0])):
    clf.fit(X_train.loc[:, fset], sy_train)
    weights = compute_sample_weight('balanced', sy_test)
    acc = clf.score(X_test.loc[:, fset], sy_test, sample_weight=weights)
    print(fset_name + ":", acc)
