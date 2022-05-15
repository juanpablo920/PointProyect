import numpy as np
import open3d as o3d
import pandas as pd
from params import ParamServer

parSer = ParamServer()

pwd_file = ""
pwd_file += parSer.prefix
pwd_file += "pointProyect/data/training/"
file = pwd_file + parSer.data_file_train

data = pd.read_csv(file, sep=" ", header=0)
Classification = np.array(data.Classification)/100.0
data = data.drop(['Classification'], axis=1)

pcd_xyz = o3d.geometry.PointCloud()
pcd_xyz.points = o3d.utility.Vector3dVector(data.to_numpy())

colors = np.zeros((data.shape[0], 3))
colors[:, 0] = Classification

pcd_xyz.colors = o3d.utility.Vector3dVector(colors)
print("datos:", len(pcd_xyz.points))

every_k_points = int(input("every_k_points:"))
lowPcd_xyz = pcd_xyz.uniform_down_sample(every_k_points)
print("low datos:", len(lowPcd_xyz.points))

name_x = (
    parSer.data_file_train[:len(parSer.data_file_train)-4]+"_low_"+str(every_k_points)+".txt")

file = pwd_file + name_x
with open(file, 'w') as f:
    f.write("X Y Z Classification\n")
    for idx in range(0, len(lowPcd_xyz.points)):
        x_tmp, y_tmp, z_tmp = lowPcd_xyz.points[idx]
        Classification_tmp = lowPcd_xyz.colors[idx][0]*100

        f.write(str(x_tmp)+" " +
                str(y_tmp)+" " +
                str(z_tmp)+" " +
                str(Classification_tmp)+"\n")
