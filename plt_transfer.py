import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


res1 = np.load("results/transfer/gr_n_css999_rs95859_nd30_ln1_d85_750250_transfer1.npy")
res2 = np.load("results/transfer/gr_n_css999_rs95859_nd30_ln1_d85_750250_transfer2.npy")

fig, ax = plt.subplots(1 ,1, figsize=(15, 5))

ax.plot(gaussian_filter1d(res1[:, 7], 4), label="SSTML with ImageNet transfer", c="red", lw=2)
ax.plot(gaussian_filter1d(res2[:, 7], 4), label="SSTML without ImageNet transfer", c="blue", lw=2)

print(np.mean(res1[:, 7]))
print(np.mean(res2[:, 7]))

ax.set_ylim(0.6, 1.0)
ax.set_xlim(0, 1000)
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("chunks")
ax.set_ylabel("BAC")


plt.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.11), fontsize=17)
plt.tight_layout()
plt.savefig("figures/transfer.png")
plt.savefig("figures/transfer.eps")