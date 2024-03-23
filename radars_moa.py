import numpy as np
import matplotlib.pyplot as plt
from utils import generate_imb_streams
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

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
# drift x stream x methods x chunks
gather = gather.reshape((2, 9, 6, 1999, 10))

# DRIFT x STREAM x METHODS x CHUNKS x METRICS
drift_scores = gather[:, :, :, :, [0, 3, 4, 5, 6, 7, 8]]
drift_scores = np.nan_to_num(drift_scores, 0)

# values, counts = np.unique(drift_scores[0, :, 5, :, 6], return_counts=True)
# print(values, counts)

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

# DRIFT x METHODS x METRICS
mean_drift_scores = np.mean(drift_scores, axis=(1, 3))[0]
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
# plt.legend(loc=(0.9, 0.9), frameon=False)

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
     
plt.title("MOA synthetic sudden drift", fontsize=17, x=0.5, y=1.07)
plt.savefig("figures/radars/radar_moa_sudden.png")
plt.savefig("figures/radars/radar_moa_sudden.eps")