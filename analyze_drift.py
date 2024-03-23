import numpy as np
from utils import generate_imb_streams
import matplotlib
from utils import DriftEvaluator
import strlearn as sl
from tabulate import tabulate


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

colors = ['silver', 'darkorange', 'seagreen', 'darkorchid', 'dodgerblue', 'red']
lws = [1.5, 1.5, 1.5 ,1.5 ,1.5 ,2]
lss = ["-", "-", "-", "-", "-", "-"]

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
# STREAM x METHODS x CHUNKS
drift_scores = drift_scores[0, :, :, :, 6]
print(drift_scores.shape)

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

drift_indxs = get_real_drfs(3000, 30).astype(int)

all_performance_loss = np.zeros((6, 20))
all_recovery_lengths = np.zeros((6, 20))
for stream_id in range(20):
    for method_id, method in enumerate(methods):
        scores = drift_scores[stream_id, method_id]
        eval = DriftEvaluator(scores.reshape(1,2999,1), drift_indxs)
        max_performance_loss = np.array(eval.get_max_performance_loss())
        recovery_lengths = np.array(eval.get_recovery_lengths())
        all_performance_loss[method_id, stream_id] = np.mean(max_performance_loss)
        all_recovery_lengths[method_id, stream_id] = np.mean(recovery_lengths)
# print(methods)
synth_performance_loss = np.mean(all_performance_loss, axis=1)
synth_recovery_length = np.mean(all_recovery_lengths, axis=1)


"""
Wilcoxon
"""

"MOA"

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
gather = gather[0, :, :, :, :, 8]
# stream x methods x chunks
gather = gather.reshape((9, 6, 1999))
moa_drift_idx = [285, 570, 857, 1142, 1427, 1713]


all_performance_loss = np.zeros((6, 9))
all_recovery_lengths = np.zeros((6, 9))
for stream_id in range(9):
    for method_id ,method in enumerate(methods):
        scores = gather[stream_id, method_id]
        eval = DriftEvaluator(scores.reshape(1,1999,1), moa_drift_idx)
        max_performance_loss = np.array(eval.get_max_performance_loss())
        recovery_lengths = np.array(eval.get_recovery_lengths())
        all_performance_loss[method_id, stream_id] = np.mean(max_performance_loss[3:])
        all_recovery_lengths[method_id, stream_id] = np.mean(recovery_lengths[:3])
        
moa_performance_loss = np.mean(all_performance_loss, axis=1)
moa_recovery_length = np.mean(all_recovery_lengths, axis=1)

first_column = np.array(["performance loss", "restoration time", "performance loss", "restoration time"]).reshape(-1, 1)

tab = np.array([synth_performance_loss, synth_recovery_length, moa_performance_loss, moa_recovery_length])
tab = np.concatenate((first_column, tab), axis=1)

print(tabulate(tab, tablefmt="latex_booktabs", floatfmt=".3f", headers=["Performance metric"] + methods))