import numpy as np
from mde import STML
from utils import generate_imb_streams
import matplotlib.pyplot as plt


streams = generate_imb_streams(1410, 5)
X, y = streams[0].get_chunk()
print(X)

X = STML(X[:1], size=(100, 100))

plt.imsave("figures/stml.png", X[0])
plt.imsave("figures/stml.eps", X[0])