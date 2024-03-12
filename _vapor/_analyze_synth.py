import numpy as np
from utils import generate_imb_streams
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

random_state = 1410
replications = 5
streams = generate_streams(random_state, replications)

methods = [
        "HF",
        # "SEA",
        # "AWE",
        # "AUE",
        "CDS",
        "NIE",
        # "WAE",
        "KUE",
        "ROSE",
        # "LearnNSE"
    ]

scores = []
scores_stml = []
scores_igtd = []
for stream in streams:
    results = np.load("results/ref_synth/%s_imb.npy" % stream)
    scores.append(results)
    
    results_stml = np.load("results/stml_synth/%s.npy" % stream)
    scores_stml.append(results_stml)
    
    # results_igtd = np.load("results/igtd_synth/%s.npy" % stream)
    # scores_igtd.append(results_igtd)
    
scores = np.array(scores).squeeze()
scores_stml = np.array(scores_stml).squeeze()
scores_igtd = np.array(scores_igtd).squeeze()
# STREAMS x METHODS x CHUNKS x METRICS
print(scores.shape)

scores_drift = [np.mean(scores[:5], axis=0), 
                np.mean(scores[5:10], axis=0), 
                np.mean(scores[10:15], axis=0)]

scores_drift_stml = [
    np.mean(scores_stml[:5], axis=0), 
                np.mean(scores_stml[5:10], axis=0), 
                np.mean(scores_stml[10:15], axis=0)
]

# scores_drift_igtd = [
#     np.mean(scores_igtd[:5], axis=0), 
#                 np.mean(scores_igtd[5:10], axis=0), 
#                 np.mean(scores_igtd[10:15], axis=0)
# ]

fig, ax = plt.subplots(3, 1, figsize=(20, 10))
ax = ax.ravel()
for drift_id , drift in enumerate(scores_drift):
    for method_id, method in enumerate(methods):
        ax[drift_id].plot(gaussian_filter1d(drift[method_id], 3), label=method)
        ax[drift_id].set_ylim(.4, 1.0)
        ax[drift_id].set_xlim(0, 2000)
        ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
        
    ax[drift_id].plot(gaussian_filter1d(scores_drift_stml[drift_id], 3), label="STML", c="red", lw=3)
    
    # ax[drift_id].plot(gaussian_filter1d(scores_drift_igtd[drift_id], 1), label="IGTD", c="blue", lw=3)

plt.legend(ncol=7)
plt.tight_layout()
plt.savefig("foo.png", dpi=100)