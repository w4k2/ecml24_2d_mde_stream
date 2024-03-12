import numpy as np
import strlearn as sl
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score as bac
from skmultiflow.trees import HoeffdingTree
from mde import STML
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# stream = sl.streams.StreamGenerator(n_informative=8, n_features=8, n_redundant=0, n_repeated=0, n_drifts=10, random_state=1)

# X, y = stream.get_chunk()
# print(X[0])
# dis = KBinsDiscretizer(3)
# X = dis.fit_transform(X).toarray().astype(int)
# print(X[0].shape)

# X_plot = STML(X, (50,50), n_cols=5)

# plt.imshow(X_plot[0])
# plt.savefig("wuj.png")



# Quantization

X, y = make_classification(n_samples=1000, n_features=8, n_informative=8, n_redundant=0, n_repeated=0, random_state=1410)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = np.digitize(X, np.linspace(0, 1, 256)).astype(int)


print(X[0])

exit()


bins, counts = np.unique(X, return_counts=True)
print(bins, counts)
exit()
# X = np.digitize(X, np.linspace(0, 1, 256))
counts, bins = np.histogram(X, bins=5)
print(bins, counts)
bins, counts = np.unique(X, return_counts=True)
# exit()
print(bins.shape, counts.shape)
plt.stairs(counts, np.arange(101))
plt.savefig("wuj.png")