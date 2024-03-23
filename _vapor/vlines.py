import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


scores = np.load("results/ref_synth/gr_n_css5_rs2417_nd30_ln1_d85_750000_imb.npy")
print(scores.shape)
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(gaussian_filter1d(scores[4, :, 6], 10), c="k")

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts 

drift_indxs = get_real_drfs(3000, 30).astype(int)
drift_indxs2 = np.load("results/drift_idx_sl_3000.npy")

ax.vlines(drift_indxs, ymin=.7, ymax=1, color="b")
ax.vlines(drift_indxs2, ymin=.7, ymax=1, color="r")

plt.tight_layout()
plt.savefig("wuj.png")