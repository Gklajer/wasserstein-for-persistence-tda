# /// script
# dependencies = [
#   "numpy",
#   "vtk",
# ]
# ///

import argparse
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def convert_to_csv(in_path: Path, out_path: Path) -> None:
    """Converts a single VTU file to a CSV file extraction Birth, Death, and PairType."""
    # Create the reader
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(in_path))
    reader.Update()

    data = reader.GetOutput()
    cell_data = data.GetCellData()

    # Helper function to get numpy array from VTK cell data
    def get_array(name):
        arr = cell_data.GetArray(name)
        if not arr:
            raise KeyError(f"Array '{name}' not found in VTU file.")
        return vtk_to_numpy(arr)

    # Extract arrays
    birth = get_array("Birth")
    persistence = get_array("Persistence")
    is_finite = get_array("IsFinite")
    pair_type = get_array("PairType")

    # Calculate death
    death = birth + persistence
    # Handle infinite points (IsFinite 0/1 to boolean)
    death[is_finite == 0] = np.inf

    # Stack into a single table
    csv_table = np.vstack((birth, death, pair_type)).T

    # Save to CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, csv_table, delimiter=",", fmt="%.17g")


def convert_vtu_dir_to_csv(input_dir: Path, output_dir: Path) -> None:
    """Recursively converts all .vtu files in a directory to .csv files in another directory."""
    vtu_files = sorted(list(input_dir.rglob("*.vtu")))

    if not vtu_files:
        print(f"No .vtu files found in {input_dir}")
        return

    print(f"Found {len(vtu_files)} files. Starting conversion...")

    for in_path in vtu_files:
        # Calculate relative path to maintain directory structure
        rel_path = in_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(".csv")

        print(f"Converting: {rel_path}")
        try:
            convert_to_csv(in_path, out_path)
        except Exception as e:
            print(f"Failed to convert {in_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VTU persistence diagrams to CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./VTU"),
        help="Directory containing .vtu files (default: ./VTU)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./CSV"),
        help="Directory where .csv files will be saved (default: ./CSV)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    convert_vtu_dir_to_csv(args.input, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
