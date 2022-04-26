import numpy as np
import open3d as o3d
import pandas as pd
from params import ParamServer

parSer = ParamServer()

file = ""
file += parSer.prefix
file += "pointProyect/data/training/"
file += parSer.data_file

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
# o3d.visualization.draw_geometries([pcd_xyz])

# every_k_points = int(input("every_k_points:"))
# lowPcd_xyz = pcd_xyz.uniform_down_sample(every_k_points)
# print("low datos:", len(lowPcd_xyz.points))
# o3d.visualization.draw_geometries([lowPcd_xyz])

every_k_points = float(input("every_k_points:"))
lowPcd_xyz, idx, lis = pcd_xyz.voxel_down_sample_and_trace(
    every_k_points, pcd_xyz.get_min_bound(), pcd_xyz.get_max_bound(), False)
print("low datos:", len(lowPcd_xyz.points))
print(idx.shape)
print(len(lis))
print(idx[:3])
print(lis[:3])
# o3d.visualization.draw_geometries([lowPcd_xyz])
