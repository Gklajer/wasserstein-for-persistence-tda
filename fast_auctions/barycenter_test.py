import wasp
import matplotlib.pyplot as plt
import numpy as np

def plot_matching(bidders, objects, off_diag, diag, fmt):
    for (i, j) in enumerate(off_diag):
        if j != 2 ** 32 - 1:
            paired = objects[j]
        else:
            x = 0.5 * bidders[i].sum()
            paired = np.array([x, x])
        line = np.array([bidders[i], paired])
        plt.plot(line[:, 0], line[:, 1], fmt)

    for (i, j) in enumerate(diag):
        if j != 2 ** 32 - 1:
            continue
        point = objects[i]
        x = 0.5 * point.sum()
        paired = np.array([x, x])
        line = np.array([point, paired])
        plt.plot(line[:, 0], line[:, 1], fmt)

diagram0 = wasp.sample_normal_diagram(10)
diagram1 = wasp.sample_normal_diagram(10)
bary, off_diag, diag = wasp.wasserstein_barycenter([diagram0, diagram1], np.array([0.5, 0.5]), 1.0)

plt.scatter(diagram0[:, 0], diagram0[:, 1], c = "r", alpha=0.5)
plot_matching(diagram0, bary, off_diag[0], diag[0], "-r")

plt.scatter(diagram1[:, 0], diagram1[:, 1], c = "g", alpha=0.5)
plot_matching(diagram1, bary, off_diag[1], diag[1], "-g")

plt.scatter(bary[:, 0], bary[:, 1], c = "b", alpha=0.5)
plt.plot([0.0, 100.0], [0.0, 100.0], c = "b")
plt.show()
