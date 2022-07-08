import os
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import time as tm

from sklearn import cluster
from params import ParamServer
from scipy.stats import norm
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class clustering:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data(self):
        print("read_data")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/data/validation/"
        file += self.parSer.data_file_valid

        data = pd.read_csv(file, sep=" ", header=0)
        self.Classification = np.array(data.Classification)
        data = data.drop(['Classification'], axis=1)

        self.pcd = data.to_numpy()

        print("datos:", self.pcd.shape)

    def cluster(self):
        silueta = []
        K = range(2, 20)
        for k in K:
            kmeans = KMeans(n_clusters = k).fit(self.pcd)
            labels = kmeans.labels_
            silueta.append(silhouette_score(self.pcd, labels, metric = 'euclidean'))
        plt.plot(K, silueta, 'bx-')
        plt.xlabel('Clústeres')
        plt.ylabel('Puntaje de la silueta')
        plt.title('Metodo de la Silueta')
        plt.show

if __name__ == '__main__':
    Cluster = clustering()
