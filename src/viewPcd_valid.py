import numpy as np
import open3d as o3d
import pandas as pd
from params import ParamServer

parSer = ParamServer()

file = ""
file += parSer.prefix
file += "pointProyect/data/validation/"
file += parSer.data_file_valid

data = pd.read_csv(file, sep=" ", header=0)
Classification = np.array(data.Classification)
data = data.drop(['Classification'], axis=1)

pcd_xyz = o3d.geometry.PointCloud()
pcd_xyz.points = o3d.utility.Vector3dVector(data.to_numpy())

pos_Tree = Classification == 16  # Tree
pos_ground = Classification == 2  # ground
pos_Model_keypoints = Classification == 8  # Model_keypoints

colors = np.zeros((data.shape[0], 3))
colors[pos_Tree, :] = [0.136, 0.556, 0.136]
colors[pos_ground, :] = [0.512, 0.256, 0]
colors[pos_Model_keypoints, :] = [0.624, 0.624, 0.624]

pcd_xyz.colors = o3d.utility.Vector3dVector(colors)
print("datos:", len(pcd_xyz.points))
o3d.visualization.draw_geometries([pcd_xyz])
