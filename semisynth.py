from strlearn.streams import SemiSyntheticStreamGenerator
from sklearn.datasets import load_iris
import strlearn as sl
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
import numpy as np
from utils import generate_semisynth_streams
from tqdm import tqdm
from skmultiflow.trees import HoeffdingTree
from scipy.ndimage import gaussian_filter1d
from mde import STML
from time import sleep


streams = generate_semisynth_streams(1410, 5)

# for stream_name in tqdm(streams.keys()):
#     stream = streams[stream_name]
#     X, y = stream.get_chunk()
#     print(X.shape[1])
#     img = STML(X[:1], size=(100, 100))
#     plt.imshow(img[0])
#     plt.savefig("wuj.png")
#     plt.close()
#     sleep(5)

# for stream_name in tqdm(streams.keys()):
#     stream = streams[stream_name]

#     eval = sl.evaluators.TestThenTrain(metrics=(bac))
#     clf = HoeffdingTree(split_criterion="hellinger")

#     eval.process(stream, [clf])
    
#     np.save("chuj/%s" % stream_name, eval.scores)

fig, ax = plt.subplots(5, 1, figsize=(15, 15))
ax = ax.ravel()

for id, stream_name in enumerate(tqdm(streams.keys())):
    results = np.load("chuj/%s.npy" % stream_name)
    results = np.array(gaussian_filter1d(results.squeeze(), 3))

    ax[id].plot(results)
    ax[id].set_ylim(0.0, 1.0)
    
plt.tight_layout()
plt.savefig("wuj.png")