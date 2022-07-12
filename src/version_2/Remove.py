import os
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time as tm
from params import ParamServer


class removeGround:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data(self):
        print("read_data")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/data/training/"
        file += self.parSer.data_file_train

        data = pd.read_csv(file, sep=" ", header=0)
        self.Classification = np.array(data.Classification)
        print(data)
        data = data.drop(['Classification'] == 3, axis=0)

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("datos:", len(self.pcd.points))
    
    def save_dps_type(self, X, Y, Z):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis"
        file += removeGround + ".txt"
        with open(file, 'a') as f:
            f.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")

if __name__ == '__main__':
    rmGround = removeGround()
    rmGround.read_data()
    rmGround.save_dps_type()
