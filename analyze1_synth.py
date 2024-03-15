import numpy as np
from utils import generate_imb_streams
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

random_state = 1410
replications = 5
streams = generate_imb_streams(random_state, replications)

metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

methods = [
        "HF",
        "CDS",
        "NIE",
        "KUE",
        "ROSE",
        "STML"
    ]

colors = ['gray', 'green', 'green', 'blue', 'blue', 'red']
lws = [1, 1, 1 ,1 ,1 ,2]
lss = ["-", "-", "--", "-", "--", "-"]

# DRIFT x STREAM x METHODS x CHUNKS x METRICS
drift_scores = np.zeros((3, 20, 5, 2999, 7))
stml_drift_scores = np.zeros((3, 20, 1, 2999, 7))
weight_scores = np.zeros((2, 30, 5, 2999, 7))
stml_weight_scores = np.zeros((2, 30, 1, 2999, 7))
ln_scores = np.zeros((2, 30, 5, 2999, 7))
stml_ln_scores = np.zeros((2, 30, 1, 2999, 7))
drift_a, drift_b, drift_c  = 0, 0 ,0
ln_a, ln_b  = 0, 0
weight_a, weight_b = 0, 0

for stream_id, stream in enumerate(streams):
    if "gr_n_css999_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        drift_scores[0, drift_a] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_drift_scores[0, drift_a] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        drift_a += 1
    if "gr_n_css5_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        drift_scores[1, drift_b] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_drift_scores[1, drift_b] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        drift_b += 1
    if "inc_n_css5_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        drift_scores[2, drift_c] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_drift_scores[2, drift_c] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        drift_c += 1
    if "ln1_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        ln_scores[0, ln_a] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_ln_scores[0, ln_a] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        ln_a += 1
    if "ln5_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        ln_scores[1, ln_b] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_ln_scores[1, ln_b] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        ln_b += 1
    if "d85_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        weight_scores[0, weight_a] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_weight_scores[0, weight_a] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        weight_a += 1
    if "d95_" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        weight_scores[1, weight_b] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_weight_scores[1, weight_b] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        weight_b += 1
        
drift_scores = np.nan_to_num(drift_scores, nan=0.0)
stml_drift_scores = np.nan_to_num(stml_drift_scores, nan=0.0)
drift_scores = np.concatenate((drift_scores, stml_drift_scores), axis=2)

ln_scores = np.nan_to_num(ln_scores, nan=0.0)
stml_ln_scores = np.nan_to_num(stml_ln_scores, nan=0.0)
ln_scores = np.concatenate((ln_scores, stml_ln_scores), axis=2)

weight_scores = np.nan_to_num(weight_scores, nan=0.0)
stml_weight_scores = np.nan_to_num(stml_weight_scores, nan=0.0)
weight_scores = np.concatenate((weight_scores, stml_weight_scores), axis=2)

"""
Drift type
"""
drift_names = ["Sudden drift", "Gradual drift", "Incremental drift"]

for metric_id ,metric in enumerate(metrics):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    ax = ax.ravel()
    for drift_id , drift in enumerate(drift_scores):
        mean_drift_scores = np.mean(drift, axis=0)
        for method_id, method in enumerate(methods):
            ax[drift_id].plot(gaussian_filter1d(mean_drift_scores[method_id, :, metric_id], 3), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])
            ax[drift_id].set_ylim(0.1, 1.0)
            ax[drift_id].set_xlim(0, 3000)
            ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
            ax[drift_id].set_title(drift_names[drift_id])
            
    plt.legend(ncol=6)
    plt.tight_layout()
    plt.savefig("figures/ex_imb/drift_%s.png" % metric, dpi=200)
    plt.close()

"""
Label noise
"""
ln_names = ["1% label noise", "5% label noise"]

for metric_id ,metric in enumerate(metrics):
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax = ax.ravel()
    for drift_id , drift in enumerate(ln_scores):
        mean_drift_scores = np.mean(drift, axis=0)
        for method_id, method in enumerate(methods):
            ax[drift_id].plot(gaussian_filter1d(mean_drift_scores[method_id, :, metric_id], 3), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])
            ax[drift_id].set_ylim(0.1, 1.0)
            ax[drift_id].set_xlim(0, 3000)
            ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
            ax[drift_id].set_title(ln_names[drift_id])
            
    plt.legend(ncol=6)
    plt.tight_layout()
    plt.savefig("figures/ex_imb/ln_%s.png" % metric, dpi=200)
    plt.close()
    
"""
Imbalance ratio
"""
imb_names = ["15% minority", "5% minority"]

for metric_id ,metric in enumerate(metrics):
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax = ax.ravel()
    for drift_id , drift in enumerate(weight_scores):
        mean_drift_scores = np.mean(drift, axis=0)
        for method_id, method in enumerate(methods):
            ax[drift_id].plot(gaussian_filter1d(mean_drift_scores[method_id, :, metric_id], 3), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])
            ax[drift_id].set_ylim(0.1, 1.0)
            ax[drift_id].set_xlim(0, 3000)
            ax[drift_id].grid(ls=":", c=(0.7, 0.7, 0.7))
            ax[drift_id].set_title(imb_names[drift_id])
            
    plt.legend(ncol=6)
    plt.tight_layout()
    plt.savefig("figures/ex_synth/d_%s.png" % metric, dpi=200)
    plt.close()