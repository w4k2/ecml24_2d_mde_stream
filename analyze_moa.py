import numpy as np
import os
from utils import moa_streams
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


drifts = ["sd", "id"]
drift_names = ["sudden", "incremental"]
generators = ["sea", "rbf", "hyp"]
replications = ["rep1", "rep2", "rep3"]
# drift x generator x replication x methods x chunks x metrics
gather = np.zeros((2, 3, 3, 6, 1999, 10))


for drift_id, drift in enumerate(drifts):
    for generator_id, generator in enumerate(generators):
        for replication_id, replication in enumerate(replications):
            scores = np.load("results/ref_moa/%s_s_%s_%s.npy" % (drift, generator, replication))
            scores_stml = np.load("results/stml_moa/%s_s_%s_%s.npy" % (drift, generator, replication))

            scores_all = np.concatenate((scores, scores_stml.reshape((1, 1999, 10))), axis=0)
            gather[drift_id, generator_id, replication_id] = scores_all

gather = np.mean(gather, axis=2)
# metrics
# drift x generator x methods x chunks x metrics
gather = gather[:, :, :, :, [0, 3, 4, 5, 6, 7, 8]]

"""
PLOTS
"""

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

for generator_id, generator in enumerate(generators):
    for metric_id, metric in enumerate(metrics):
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        ax = ax.ravel()
        for drift_id, drift in enumerate(drifts):
            drift_scores = gather[drift_id, generator_id, :, :, metric_id]
            for method_id, method in enumerate(methods):
                ax[drift_id].plot(gaussian_filter1d(drift_scores[method_id], 3), label=method, ls=lss[method_id], lw=lws[method_id], c=colors[method_id])
                ax[drift_id].set_title("%s %s" % (generator, drift_names[drift_id]))
                ax[drift_id].spines[['right', 'top']].set_visible(False)
                ax[drift_id].set_ylim(0.5, 1.0)
                ax[drift_id].set_xlabel("chunks")
                ax[drift_id].set_ylabel("BAC")
                ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
                ax[drift_id].set_xlim(0, 2000)
                
        ax[0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.25), fontsize=17)
        plt.tight_layout()
        plt.savefig("figures/ex_moa/%s_%s.png" % (generator, metric))
        plt.savefig("figures/ex_moa/%s_%s.eps" % (generator, metric))
        plt.close()
        
"""
Mean for drifts
"""
# drift x methods x chunks x metrics
gather_mean_drifts = np.mean(gather, axis=1)

for metric_id, metric in enumerate(metrics):
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax = ax.ravel()
    for drift_id, drift in enumerate(drifts):
        mean_drift_scores = gather_mean_drifts[drift_id, :, :, metric_id]
        for method_id, method in enumerate(methods):
            ax[drift_id].plot(gaussian_filter1d(mean_drift_scores[method_id], 5), label=method, ls=lss[method_id], lw=lws[method_id], c=colors[method_id])
            ax[drift_id].set_title("MOA %s" % (drift_names[drift_id]))
            ax[drift_id].spines[['right', 'top']].set_visible(False)
            ax[drift_id].set_ylim(0.5, 1.0)
            ax[drift_id].set_xlabel("chunks")
            ax[drift_id].set_ylabel("BAC")
            ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
            ax[drift_id].set_xlim(0, 2000)
            
    ax[0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.35), fontsize=17)
    plt.tight_layout()
    plt.savefig("figures/ex_moa/moa_drift_%s.png" % (metric))
    plt.savefig("figures/ex_moa/moa_drift_%s.eps" % (metric))
    plt.close()