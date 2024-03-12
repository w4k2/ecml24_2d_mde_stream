import numpy as np
from utils import realstreams
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

random_state = 1410
streams = realstreams()
n_chunks = [265, 359]

methods = [
        "HF",
        # "SEA",
        # "AWE",
        # "AUE",
        "CDS",
        "NIE",
        "WAE",
        "KUE",
        "ROSE",
        # "LearnNSE"
    ]

scores = []
scores_stml = []
scores_igtd = []
for stream in streams:
    results = np.load("results/ref_real/%s.npy" % stream)
    scores.append(results.squeeze())
    
    results_stml = np.load("results/stml_real/%s.npy" % stream)
    scores_stml.append(results_stml.squeeze())
    
    results_igtd = np.load("results/igtd_real/%s.npy" % stream)
    scores_igtd.append(results_igtd.squeeze())

# scores_stml = np.array(scores_stml).squeeze()
# STREAMS x METHODS x CHUNKS x METRICS

fig, ax = plt.subplots(len(streams), 1, figsize=(20, 10))
ax = ax.ravel()
for stream_id , stream in enumerate(streams):
    for method_id, method in enumerate(methods):
        ax[stream_id].plot(gaussian_filter1d(scores[stream_id][method_id], 2), label=method)
        ax[stream_id].set_ylim(.4, 1.0)
        ax[stream_id].set_xlim(0, n_chunks[stream_id])
        ax[stream_id].grid(ls=":", c=(0.7, 0.7, 0.7))
    ax[stream_id].plot(gaussian_filter1d(scores_stml[stream_id], 2), label="STML", c="red", lw=3)
    ax[stream_id].plot(gaussian_filter1d(scores_igtd[stream_id], 2), label="IGTD", c="blue", lw=3)

plt.legend(ncol=8)
plt.tight_layout()
plt.savefig("bar.png", dpi=300)