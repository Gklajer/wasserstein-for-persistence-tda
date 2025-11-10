from paraview import simple
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_to_csv(in_path, out_path):
    reader = simple.OpenDataFile(in_path)
    reader.UpdatePipeline()
    data = simple.servermanager.Fetch(reader)

    cells = data.GetCells()
    offsets = np.array(cells.GetOffsetsArray())
    connectivity = np.array(cells.GetConnectivityArray())

    #cell_sizes = np.ediff1d(np.sort(offsets))
    #assert np.all(cell_sizes == 2)

    #vertex_0 = connectivity[offsets[:-1]]
    #vertex_1 = connectivity[offsets[:-1] + 1]

    birth = np.array(data.GetCellData().GetArray("Birth"))
    persistence = np.array(data.GetCellData().GetArray("Persistence"))
    death = birth + persistence
    is_finite = np.array(data.GetCellData().GetArray("IsFinite"))
    death[np.logical_not(is_finite)] = np.inf
    ty = np.array(data.GetCellData().GetArray("PairType"))

    #inf_indices = np.where(np.logical_not(is_finite))[0]

    #plt.scatter(birth, death, c = ty)
    #plt.scatter(birth[inf_indices], death[inf_indices], color = "#f00")
    #plt.show()

    csv_table = np.vstack((birth, death, ty)).T
    np.savetxt(out_path, csv_table, delimiter=',', fmt='%.17g')

def convert_vtu_dir_to_csv(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".vtu"):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".csv")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                convert_to_csv(in_path, out_path)

convert_vtu_dir_to_csv("./VTU", "./CSV")    

