import numpy as np
from utils import generate_imb_streams
import matplotlib
from utils import DriftEvaluator
import strlearn as sl
from tabulate import tabulate
from scipy.stats import rankdata, ranksums


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

random_state = 1410
replications = 5
streams = generate_imb_streams(random_state, replications)

filtr = 10

metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

methods = [
        "HF",
        "CDS",
        "NIE",
        "KUE",
        "ROSE",
        "SSTML"
    ]

drifts = ["sudden", "gradual", "incremental"]

"""
SYNTH
"""

# DRIFT x STREAM x METHODS x CHUNKS x METRICS
drift_scores = np.zeros((3, 20, 5, 2999, 7))
stml_drift_scores = np.zeros((3, 20, 1, 2999, 7))
drift_a, drift_b, drift_c  = 0, 0 ,0

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
        
drift_scores = np.nan_to_num(drift_scores, nan=0.0)
stml_drift_scores = np.nan_to_num(stml_drift_scores, nan=0.0)
# DRIFT x STREAM x METHODS x CHUNKS x METRICS
drift_scores = np.concatenate((drift_scores, stml_drift_scores), axis=2)
# Sudden and BAC only
# DRIFT x STREAM x METHODS x CHUNKS
gathered_scores = drift_scores[:, :, :, :, 6]
# DRIFT x STREAM x METHODS
gathered_scores = np.mean(gathered_scores, axis=3)
print(gathered_scores.shape)

t = []
length = len(methods)
alpha = .05
for drift_id ,drift in enumerate(drifts):
    drift_scores = gathered_scores[drift_id]
    drift_ranks = rankdata(drift_scores, axis=1)
    mean_drift_ranks = np.mean(drift_ranks, axis=0)
    
    s = np.zeros((length, length))
    p = np.zeros((length, length))
    
    for i in range(length):
        for j in range(length):
            s[i, j], p[i, j] = ranksums(drift_ranks.T[i], drift_ranks.T[j])
    
    _ = np.where((p < alpha) * (s > 0))
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(length)]
    
    t.append(["%s" % drift] + ["%.3f" % v for v in mean_drift_ranks])
    t.append([''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                             for c in conclusions])
    
print(tabulate(t, headers=["Drift type"] + methods, tablefmt="latex_booktabs"))


"""
MOA
"""

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


# drift x generator x replication x methods x chunks x metrics
# sd and BAC only
# generator x replication x methods x chunks
gather = gather[:, :, :, :, :, 8]
# drift x stream x methods x chunks
gather = gather.reshape((2, 9, 6, 1999))

gathered_scores = np.mean(gather[:, :, :, :], axis=3)
print(gathered_scores.shape)

t = []
length = len(methods)
alpha = .05
for drift_id ,drift in enumerate(drifts):
    drift_scores = gathered_scores[drift_id]
    drift_ranks = rankdata(drift_scores, axis=1)
    mean_drift_ranks = np.mean(drift_ranks, axis=0)
    
    s = np.zeros((length, length))
    p = np.zeros((length, length))
    
    for i in range(length):
        for j in range(length):
            s[i, j], p[i, j] = ranksums(drift_ranks.T[i], drift_ranks.T[j])
    
    _ = np.where((p < alpha) * (s > 0))
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(length)]
    
    t.append(["%s" % drift_names[drift_id]] + ["%.3f" % v for v in mean_drift_ranks])
    t.append([''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                             for c in conclusions])
    
print(tabulate(t, headers=["Drift type"] + methods, tablefmt="latex_booktabs"))