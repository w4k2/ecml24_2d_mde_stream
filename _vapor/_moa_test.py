import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


res1 = np.load("results/transfer/gr_n_css999_rs95859_nd10_ln1_d85_250250_transfer1.npy")
res2 = np.load("results/transfer/gr_n_css999_rs95859_nd10_ln1_d85_250250_transfer2.npy")


plt.plot(gaussian_filter1d(res1[:, 7], 5), label="YES")
plt.plot(gaussian_filter1d(res2[:, 7], 5), label="NO")

# results1 = np.load("results/ref_moa/id_s_sea_r1_s_sea_r2.npy")
# results2 = np.load("results/stml_moa/id_s_sea_r1_s_sea_r2.npy")
# print(results1.shape)
# print(results2.shape)

# methods = [
#         "HF",
#         "CDS",
#         "NIE",
#         "KUE",
#         "ROSE",
#         # "STML"
#     ]

# colors = ['gray', 'green', 'green', 'blue', 'blue', 'red']
# lws = [1, 1, 1 ,1 ,1 ,2]
# lss = ["-", "-", "--", "-", "--", "-"]

# for m in range(5):
#     plt.plot(gaussian_filter1d(results1[m, :, 8], 6), c=colors[m], ls=lss[m], label=methods[m])
    
plt.legend()
# plt.plot(gaussian_filter1d(results2[:, 8], 6), c="red", lw=2, label="STML")
plt.savefig("wuj.png")