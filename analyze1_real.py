import numpy as np
from utils import realstreams
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

random_state = 1410
streams = realstreams()
n_chunks = [265, 359]

metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

methods = [
        "HF",
        "CDS",
        "NIE",
        "KUE",
        "ROSE",
        "STML"
    ]

colors = ['silver', 'darkorange', 'seagreen', 'darkorchid', 'dodgerblue', 'red']
lws = [1.5, 1.5, 1.5 ,1.5 ,1.5 ,2]
lss = ["-", "-", "-", "-", "-", "-"]

scores_streams = []
for stream in streams:
    results = np.load("results/ref_real/%s.npy" % stream)
    results_stml = np.load("results/stml_real/%s_.npy" % stream)
    scores_stream = np.concatenate((results[: ,:, [0, 3, 4, 5, 6, 7, 8]], results_stml.reshape((1, results_stml.shape[0], results_stml.shape[1]))[:, :, [0, 3, 4, 5, 6, 7, 8]]), axis=0)
    # print(scores_stream.shape)
    scores_streams.append(scores_stream)
# exit()
# STREAMS x METHODS x CHUNKS x METRICS

for metric_id, metric in enumerate(metrics):
    fig, ax = plt.subplots(len(streams), 1, figsize=(15, 10))
    ax = ax.ravel()
    for stream_id , stream in enumerate(streams):
        for method_id, method in enumerate(methods):
            ax[stream_id].plot(gaussian_filter1d(scores_streams[stream_id][method_id, :, metric_id], 2), label=method, ls=lss[method_id], lw=lws[method_id], c=colors[method_id])
            ax[stream_id].set_xlim(0, n_chunks[stream_id])
            ax[stream_id].grid(ls=":", c=(0.7, 0.7, 0.7))
            ax[stream_id].spines[['right', 'top']].set_visible(False)
            ax[stream_id].set_ylim(0.4, 1.0)
            ax[stream_id].set_xlabel("chunks")
            ax[stream_id].set_ylabel("BAC")
            ax[stream_id].set_title("%s" % (stream))

    ax[0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.25), fontsize=17)
    plt.tight_layout()
    plt.savefig("figures/ex_real/real_%s.png" % (metric), dpi=300)
    plt.savefig("figures/ex_real/real_%s.eps" % (metric))
    plt.close()