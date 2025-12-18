# wasserstein-for-persistence-tda

`wasp` is a Rust re-implementation of the [Wasserstein representation for Persistence diagrams](https://arxiv.org/abs/2304.14852) paper. The original C++ implementation is available in [this repository](https://github.com/Keanu-Sisouk/W2-PD-Dict).

## Getting Started

### Prerequisites

1. **Rust**: Install the Rust toolchain from [rustup.rs](https://rustup.rs/).
2. **Python**: Python 3.8+ is recommended.
3. **uv** (Recommended): A fast Python package manager. Install with `brew install uv` or see [uv documentation](https://github.com/astral-sh/uv).

### Installation & Build

1. Clone the repository:

    ```bash
    git clone https://github.com/Gklajer/wasserstein-for-persistence-tda.git
    cd wasserstein-for-persistence-tda
    ```

2. Sync the environment (creates venv and installs all dependencies):

    ```bash
    uv sync
    ```

### Running the Code

After building the extension, you can run the provided Python scripts from the `examples/` directory:

- **Barycenter Test**: Visualizes the Wasserstein barycenter of two diagrams.

  ```bash
  uv run examples/barycenter_test.py
  ```

- **Matching Test**: Visualizes the optimal matching between two diagrams.

  ```bash
  uv run examples/test_maching.py
  ```

- **Benchmark**: Runs performance benchmarks.

  ```bash
  uv run examples/benchmark.py
  ```

## Data Preparation

To convert persistence diagrams from VTU format (e.g., from ParaView/TTK) to the CSV format used by this project, you can use the provided conversion script. It handles Birth, Death (including infinite values), and PairType extraction.

```bash
uv run scripts/vtu_to_csv.py --input data/VTU --output data/CSV
```

## References

The following additional papers were used in the implementation:

- [Progressive Wasserstein Barycenters of Persistence Diagrams](https://arxiv.org/pdf/1907.04565)
- [Geometry Helps to Compare Persistence Diagrams](https://arxiv.org/pdf/1606.03357)
- [Multidimensional Binary Search Trees Used for Associative Searching](https://dl.acm.org/doi/10.1145/350044.350055)
