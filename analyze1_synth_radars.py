import numpy as np
import matplotlib.pyplot as plt
from utils import generate_imb_streams


random_state = 1410
replications = 5
streams = generate_imb_streams(random_state, replications)

metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

drift_names = ["Sudden drift", "Gradual drift", "Incremental drift"]
# DRIFT x STREAM x METHODS x CHUNKS x METRICS
drift_scores = np.zeros((3, 20, 5, 2999, 7))
stml_drift_scores = np.zeros((3, 20, 1, 2999, 7))
a, b, c  = 0, 0 ,0
for stream_id, stream in enumerate(streams):
    if "gr_n_css999" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        drift_scores[0, a] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_drift_scores[0, a] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        a += 1
    if "gr_n_css5" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        drift_scores[1, b] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_drift_scores[1, b] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        b += 1
    if "inc_n_css5" in str(stream) and "ln15_" not in str(stream):
        results = np.load("results/ref_synth/%s_imb.npy" % stream)
        drift_scores[2, c] = results
        results = np.load("results/stml_synth/%s_imb.npy" % stream)
        stml_drift_scores[2, c] = results[:, [0, 1, 3, 4, 5, 6, 7]]
        c += 1
drift_scores = np.nan_to_num(drift_scores, nan=0.0)

stml_drift_scores = np.nan_to_num(stml_drift_scores, nan=0.0)
drift_scores = np.concatenate((drift_scores, stml_drift_scores), axis=2)

drift_scores = drift_scores[:, :, :, 1000:]

# values, counts = np.unique(drift_scores[0, :, 5, :, 6], return_counts=True)
# print(values, counts)

metrics=["Recall", "Precision", "Specificity", "F1", "Gmean", "Gmeans", "BAC"]

methods = [
        "HF",
        "CDS",
        "NIE",
        "KUE",
        "ROSE",
        "STML"
    ]

colors = ['gray', 'green', 'green', 'blue', 'blue', 'red']
lws = [2, 2, 2 ,2 ,2 ,3]
lss = ["-", "-", "--", "-", "--", "-"]

# DRIFT x METHODS x METRICS
mean_drift_scores = np.mean(drift_scores, axis=(1, 3))[2]
std_drift_scores = np.mean(drift_scores, axis=(1, 3))[2]

mean_drift_scores = np.concatenate((mean_drift_scores, mean_drift_scores[:,:1]), axis=1)
std_drift_scores = np.concatenate((std_drift_scores, std_drift_scores[:,:1]), axis=1)

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(metrics)+1)
plt.figure(figsize=(7, 7))
ax = plt.subplot(polar=True)

for method_id, method in enumerate(methods):
    # print(method)
    m = mean_drift_scores[method_id]
    s = std_drift_scores[method_id]
    plt.plot(label_loc, m, label=method, c=colors[method_id], lw=lws[method_id], ls=lss[method_id])
    # plt.fill_between(label_loc, m-s, m+s, color=colors[method_id], alpha=0.2)

ax = plt.gca()
ax.spines['polar'].set_visible(False)
ax.spines['start'].set_visible(False)
ax.spines['end'].set_visible(False)
ax.spines['inner'].set_visible(False)

plt.ylim(0,1)

gpoints = np.linspace(0,1,6)
plt.gca().set_yticks(gpoints)
plt.legend(loc=(0.9, 0.9), frameon=False)

ax.grid(lw=0)
ax.set_xticks(label_loc[:-1])
ax.set_xticklabels([])

gc = {
    'c':'#999',
    'lw': 1,
    'ls': ':'
}
for loc, met in zip(label_loc[:-1], metrics):
    # print(loc,met)
    ax.plot([loc,loc],[0,1], **gc)
ax.plot(np.linspace(0,2*np.pi,100), np.zeros(100), **gc)
ax.plot(np.linspace(0,2*np.pi,100), np.ones(100), **gc)

for gpoint in gpoints:
    ax.plot(np.linspace(0,2*np.pi,100), 
        np.ones(100) * gpoint, **gc)
    
    
step = np.pi*1.9/(len(metrics)-1)
for llo, lla in zip(label_loc*step, metrics):
     a = np.rad2deg(llo+np.pi/2) if llo > np.pi else np.rad2deg(llo-np.pi/2)
    #  print(a)
     ax.text(llo, 1.05, lla, rotation=a, ha='center', va='center',weight="bold")
    
plt.savefig("wuj.png")



