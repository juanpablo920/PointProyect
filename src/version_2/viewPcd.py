import numpy as np
import open3d as o3d
import pandas as pd
from version_2.params import ParamServer

parSer = ParamServer()

file = ""
file += parSer.prefix

print("Opcion_1: training")
print("Opcion_2: validation")
print("Opcion_3: results")
print("")

opcion = input("opcion: ")

if opcion == "1":
    file += "pointProyect/data/training/"
    file += parSer.data_file_train
elif opcion == "2":
    file += "pointProyect/data/validation/"
    file += parSer.data_file_valid
elif opcion == "3":
    file += "pointProyect/data/results/"
    file += "clf_" + parSer.data_file_valid
else:
    print("="*10)
    print("no es una opcion '{opcion}'")
    exit()

data = pd.read_csv(file, sep=" ", header=0)
Classification = np.array(data.Classification)
data = data.drop(['Classification'], axis=1)

pcd_xyz = o3d.geometry.PointCloud()
pcd_xyz.points = o3d.utility.Vector3dVector(data.to_numpy())

pos_Tree = Classification == 16  # Tree
pos_ground = Classification == 2  # ground

colors = np.zeros((data.shape[0], 3))
colors[pos_Tree, :] = [0.136, 0.556, 0.136]
colors[pos_ground, :] = [0.512, 0.256, 0]

pcd_xyz.colors = o3d.utility.Vector3dVector(colors)
print("datos:", len(pcd_xyz.points))
o3d.visualization.draw_geometries([pcd_xyz])
