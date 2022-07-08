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

        x1 =np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
        x2 =np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
        plt.plot()
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.title( 'Conjunto de Datos')
        plt.scatter(x1, x2)
        plt.show()

        X = np.array(list(zip(x1, x2))).reshape(len(x1),2)
        
        print("cluster")
        silueta = []
        for k in range(2, 11):
            print("K: ",k)
            kmeans = KMeans(n_clusters = k).fit(self.pcd)
            labels = kmeans.labels_
            print(labels)
            silueta.append(silhouette_score(self.pcd, labels, metric = 'euclidean'))
            print(silueta)
           
        plt.plot(range(2, 11), silueta, 'bx-')
        plt.xlabel('Clústeres')
        plt.ylabel('Puntaje de la silueta')
        plt.title('Metodo de la Silueta')
        plt.savefig("Silueta_vs_Clases.png")

if __name__ == '__main__':
    Cluster = clustering()
    Cluster.read_data()
    Cluster.cluster()
