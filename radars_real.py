import numpy as np
import matplotlib.pyplot as plt
from utils import realstreams


metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

streams = realstreams()

scores_streams = []
for stream in streams:
    results = np.load("results/ref_real/%s.npy" % stream)
    results_stml = np.load("results/stml_real/%s_.npy" % stream)
    scores_stream = np.concatenate((results[: ,:, [0, 3, 4, 5, 6, 7, 8]], results_stml.reshape((1, results_stml.shape[0], results_stml.shape[1]))[:, :, [0, 3, 4, 5, 6, 7, 8]]), axis=0)
    # print(scores_stream.shape)
    scores_streams.append(scores_stream)

metrics=["Recall", "Precision", "Specificity", "F1", "Gmean", "Gmean$_s$", "BAC"]

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

scores = scores_streams[0]

# DRIFT x METHODS x METRICS
mean_drift_scores = np.mean(scores, axis=(1))
# std_drift_scores = np.mean(drift_scores, axis=(1, 3))[2]

mean_drift_scores = np.concatenate((mean_drift_scores, mean_drift_scores[:,:1]), axis=1)
# std_drift_scores = np.concatenate((std_drift_scores, std_drift_scores[:,:1]), axis=1)

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(metrics)+1)
plt.figure(figsize=(7, 7))
ax = plt.subplot(polar=True)

for method_id, method in enumerate(methods):
    # print(method)
    m = mean_drift_scores[method_id]
    # s = std_drift_scores[method_id]
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
# plt.legend(loc="lower center", frameon=False, ncol=6)

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