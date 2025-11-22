import wasp
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark(n):
    count = 1 #should be 10 to reproduce the experiments of the paper

    pairs = []
    for _ in range(count):
        pairs.append([wasp.sample_normal_diagram(n), wasp.sample_normal_diagram(n)])
    
    start = time.time()
    for [a, b] in pairs:
        _, _, _ = wasp.wasserstein_distance(a, b, 0.01)
    end = time.time()

    return (end - start) / count

test_values = np.arange(1000, 10500, 500)[::-1]
print(test_values)
times = []

for n in test_values:
    print("Benchmarking n = " + str(n) + "...")
    times.append(benchmark(n))

plt.figure(figsize=(6,4))

n_scaled = np.array(test_values) / 1e4

plt.plot(n_scaled, times, marker="o")

plt.yscale("log")
plt.xlabel(r"$n \cdot 10^4$")
plt.ylabel("Seconds")

plt.tight_layout()
plt.savefig("benchmark.png", dpi=300)
plt.close()