import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wasp


def benchmark(n: int, count: int = 10) -> list[float]:
    """
    Benchmarks the Wasserstein distance calculation for diagrams with n points.

    Args:
        n: Number of points in each persistence diagram.
        count: Number of repetitions to average over.

    Returns:
        List of times in seconds for each calculation.
    """
    results = []
    delta = np.sqrt(1.0 + 0.01) - 1.0

    for _ in range(count):
        # Generate diagrams inside the loop to avoid memory overhead for large count
        a = wasp.sample_normal_diagram(n)
        b = wasp.sample_normal_diagram(n)

        start = time.time()
        _ = wasp.wasserstein_distance(a, b, delta)
        end = time.time()
        results.append(end - start)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark wasp performance.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark",
        help="Output path prefix (default: results/benchmark)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of runs to average (default: 10)",
    )
    args = parser.parse_args()

    # Ensure output directory exists (Best Practice: store artifacts in a dedicated, ignored folder)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Test values for n (number of points)
    test_values = np.arange(1000, 10500, 500)[::-1]
    print(f"Test values for n: {test_values}")

    all_times = []
    for n in test_values:
        print(f"Benchmarking n = {n}...")
        times_n = benchmark(int(n), count=args.count)
        all_times.append(times_n)

    all_times_arr = [np.array(t) for t in all_times]
    means = np.array([np.mean(t) for t in all_times_arr])
    mins = np.array([np.min(t) for t in all_times_arr])
    maxes = np.array([np.max(t) for t in all_times_arr])

    # Plotting
    plt.style.use("seaborn-v0_8-muted")
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Plot the actual measured times with shaded area (Min-Mean-Max)
    # Merging handles: we keep the handle for the line, and the fill_between provides context
    (line_handle,) = ax.plot(
        test_values,
        means,
        "o-",
        label=f"Measured Performance ({args.count} runs, min-mean-max)",
        linewidth=2.5,
        markersize=8,
        color="#2E86C1",
    )
    ax.fill_between(test_values, mins, maxes, color="#2E86C1", alpha=0.15, zorder=2)

    # 2. Plot the expected O(n^2) scaling curve
    expected_scaling = test_values**2
    ratio = means[-1] / expected_scaling[-1]
    expected_values = expected_scaling * ratio

    (expected_handle,) = ax.plot(
        test_values,
        expected_values,
        "--",
        label=r"Expected Scaling $O(n^2)$",
        color="#E67E22",
        alpha=0.8,
        linewidth=2,
    )

    # 3. Beautification
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # X-axis tick formatting (e.g., 2k, 4k, 6k)
    def format_ticks(x, pos):
        if x >= 1000:
            return f"{int(x/1000)}k"
        return str(int(x))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ticks = [1000, 2000, 4000, 6000, 8000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_ticks(t, None) for t in ticks], fontsize=14)

    ax.set_xlabel(r"Number of Points ($n$)", fontsize=18, labelpad=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=18, labelpad=12)
    ax.tick_params(axis="y", labelsize=14)

    # Legend outside on top, centered, 1 column chosen for clarity with long text
    ax.legend(
        handles=[line_handle, expected_handle],
        fontsize=16,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        handletextpad=0.8,
    )

    # Add a thin border/frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.3)

    plt.tight_layout()

    # Save as SVG and PNG
    plt.savefig(f"{args.output}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")

    plt.close()
    print(
        f"Benchmark completed. Results saved to {args.output}.svg and {args.output}.png"
    )


if __name__ == "__main__":
    main()
