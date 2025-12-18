import wasp
import matplotlib.pyplot as plt
import numpy as np

diagram0 = wasp.sample_normal_diagram(100)
diagram1 = wasp.sample_normal_diagram(110)
off_diag, diag, _ = wasp.wasserstein_distance(diagram0, diagram1, 0.01)

plt.plot([0.0, 100.0], [0.0, 100.0], "-r")
for (i, j) in enumerate(off_diag):
    if j != 2 ** 32 - 1:
        paired = diagram1[j]
    else:
        x = 0.5 * diagram0[i].sum()
        paired = np.array([x, x])
    line = np.array([diagram0[i], paired])
    plt.plot(line[:, 0], line[:, 1], "-b")

for (i, j) in enumerate(diag):
    if j != 2 ** 32 - 1:
        continue
    point = diagram1[i]
    x = 0.5 * point.sum()
    paired = np.array([x, x])
    line = np.array([point, paired])
    plt.plot(line[:, 0], line[:, 1], "-g")

plt.show()
