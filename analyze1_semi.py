import numpy as np
from utils import generate_semisynth_streams
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


datasets = [
    "popfailures", 
    "ecoli-0-1-4-6-vs-5", 
    "glass5", 
    "yeast6",
    "spectfheart"
    ]
interpolations = [
    "nearest", 
    "linear"
    ]
drift_name = [
    "sudden",
    "incremental"
]
seeds = [2417, 22389, 55694, 79760, 95859]

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

# DATASET x DRIFT x REPLICATION x METHODS x CHUNKS x METRICS
scores = np.zeros((5, 2, 5, 5, 1999, 7))
scores_stml = np.zeros((5, 2, 5, 1, 1999, 7))

for dataset_id, dataset in enumerate(datasets):
    for drift_id, drift in enumerate(interpolations):
        for replication_id, replication in enumerate(seeds):
            results = np.load("results/ref_semi/%s_%s_%i.npy" % (dataset, drift, replication))
            scores[dataset_id, drift_id, replication_id] = results[:, :, [0, 2, 4, 5, 6, 7, 8]]
            
            results_stml = np.load("results/stml_semi/%s_%s_%i.npy" % (dataset, drift, replication))
            scores_stml[dataset_id, drift_id, replication_id, 0] = results_stml[:, [0, 2, 4, 5, 6, 7, 8]]
        
scores = np.nan_to_num(scores, nan=0.0)
scores_stml = np.nan_to_num(scores_stml, nan=0.0)
scores = np.concatenate((scores, scores_stml), axis=3)

# DATASET x DRIFT x METHODS x CHUNKS x METRICS
mean_scores = np.mean(scores, axis=2)
print(mean_scores.shape)

for dataset_id, dataset in enumerate(datasets):
    for metric_id ,metric in enumerate(metrics):
        fig, ax = plt.subplots(2, 1, figsize=(9, 12))
        ax = ax.ravel()
        for drift_id , drift in enumerate(interpolations):
            drift_scores = mean_scores[dataset_id, drift_id, :, :, metric_id]
            for method_id, method in enumerate(methods):
                ax[drift_id].plot(gaussian_filter1d(drift_scores[method_id], 10), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])
                ax[drift_id].set_xlim(0, 2000)
                ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
                ax[drift_id].set_title("%s %s" % (dataset, drift_name[drift_id]))
                ax[drift_id].spines[['right', 'top']].set_visible(False)
                ax[drift_id].set_ylim(0.5, 1.0)
                ax[drift_id].set_xlabel("chunks")
                ax[drift_id].set_ylabel("BAC")
            
        # ax[0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.25), fontsize=17)
        plt.tight_layout()
        plt.savefig("figures/ex_semi/%s_%s.png" % (dataset, metric), dpi=200)
        plt.savefig("figures/ex_semi/%s_%s.eps" % (dataset, metric))
        plt.close()
        
"""
Mean drift
"""

#DRIFT x METHODS x CHUNKS x METRICS
mean_drift_scores = np.mean(mean_scores, axis=0)

for metric_id ,metric in enumerate(metrics):
    fig, ax = plt.subplots(2, 1, figsize=(9, 12))
    ax = ax.ravel()
    for drift_id , drift in enumerate(interpolations):
        drift_scores = mean_drift_scores[drift_id, :, :, metric_id]
        for method_id, method in enumerate(methods):
            ax[drift_id].plot(gaussian_filter1d(drift_scores[method_id], 10), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])
            ax[drift_id].set_xlim(0, 2000)
            ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
            ax[drift_id].set_title("Semi-synthetic %s" % (drift_name[drift_id]))
            ax[drift_id].spines[['right', 'top']].set_visible(False)
            ax[drift_id].set_ylim(0.5, 1.0)
            ax[drift_id].set_xlabel("chunks")
            ax[drift_id].set_ylabel("BAC")
        
    # ax[0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.25), fontsize=17)
    plt.tight_layout()
    plt.savefig("figures/ex_semi/semi_drift_%s.png" % (metric), dpi=200)
    plt.savefig("figures/ex_semi/semi_drift_%s.eps" % (metric))
    plt.close()