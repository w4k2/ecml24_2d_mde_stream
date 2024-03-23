import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

time_ref_8 = np.load("results/time/time_ref_8.npy")
time_stml_8 = np.load("results/time/time_stml_8.npy")
time_ref_16 = np.load("results/time/time_ref_16.npy")
time_stml_16 = np.load("results/time/time_stml_16.npy")
time_ref_32 = np.load("results/time/time_ref_32.npy")
time_stml_32 = np.load("results/time/time_stml_32.npy")
time_ref_64 = np.load("results/time/time_ref_64.npy")
time_stml_64 = np.load("results/time/time_stml_64.npy")

time_ref = np.array([time_ref_8, time_ref_16, time_ref_32, time_ref_64])
time_stml = np.array([time_stml_8, time_stml_16, time_stml_32, time_stml_64])

time = np.concatenate((time_ref, time_stml), axis=1)
mean_time = np.mean(time[:, :, 10:], axis=2)
std_time = np.std(time[:, :, 10:], axis=2)

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

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

for method_id, method in enumerate(methods):
    ax[0].plot([0, 16, 32, 64], mean_time[:, method_id], ls=lss[method_id], lw=lws[method_id], c=colors[method_id], label=method)
    
    # ax[0].fill_between(
    #        [0, 16, 32, 64],
    #         mean_time[:, method_id] + std_time[:, method_id],
    #         mean_time[:, method_id] - std_time[:, method_id], 
    #         ls=lss[method_id], lw=lws[method_id], color=colors[method_id],
    #         alpha=0.2,
    #     )
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set_ylim(0.0, 6.5)
    ax[0].set_xlim(0, 64)
    ax[0].set_xlabel("#features")
    ax[0].set_xticks([0, 16, 32, 64], ["8", "16", "32", "64"])
    ax[0].set_yticks(np.arange(0.0, 7.0, 0.5), [str(i) for i in np.arange(0.0, 7.0, 0.5)])
    ax[0].set_ylabel("Mean chunk processing time [s]")
    ax[0].grid(ls=":", c=(0.7, 0.7, 0.7))
    
    # ax.set_title()


ax[1].bar(methods, np.mean(mean_time, axis=0), color=colors)
ax[1].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[1].set_xlabel("method")
ax[1].set_ylabel("Mean processing time for all features [s]")
ax[1].set_ylim(0, 3.5)
ax[1].spines[['right', 'top']].set_visible(False)
# fig.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.06), fontsize=17)
plt.tight_layout()
plt.savefig("figures/time.png")
plt.savefig("figures/time.eps")