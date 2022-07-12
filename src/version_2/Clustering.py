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
        file += "pointProyect/data/training/"
        file += self.parSer.data_file_train

        data = pd.read_csv(file, sep=" ", header=0)
        self.Classification = np.array(data.Classification)
        
        indices = data[data['Classification'] == 2].index
        data = data.drop(indices)

        print("Hola")

        self.pcd = data.to_numpy()
        print(self.pcd)

        print("datos:", self.pcd.shape)

    def cluster(self):

        # x1 =np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
        # x2 =np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
        # plt.plot()
        # plt.xlim([0,10])
        # plt.ylim([0,10])
        # plt.title( 'Conjunto de Datos')
        # plt.scatter(x1, x2)
        # plt.show()

        #X = np.array(list(zip(x1, x2))).reshape(len(x1),2)

        print("cluster")
        silueta = []
        for k in range(2, 50):
            print("K: ", k)
            kmeans = KMeans(n_clusters=k).fit(self.pcd)
            labels = kmeans.labels_
            print(labels)
            silueta.append(silhouette_score(
                self.pcd, labels, metric='euclidean'))
            print(silueta)

        silueta = [0.4600763203672339, 0.4791886811848334, 0.45641667422853693, 0.43899880968732624, 0.472847239005536, 0.4708187515426087, 0.46530791591702203, 0.4930105134753089, 0.5063025823537268, 0.5166899805704104, 0.5181851286456105, 0.5279329279863526, 0.5342069772920724, 0.5382661258559718, 0.5414265414179411, 0.5355857779002761, 0.5129431245523784, 0.5197695411559515, 0.5091491017005373, 0.5080775124640468, 0.5061471482411569, 0.4894572036084337, 0.473635111504707, 0.4767245048442456,
                   0.47744463833828826, 0.46770420517806266, 0.48594549618107924, 0.46000095667299895, 0.466268358464164, 0.4459063003377373, 0.4616430328141441, 0.45718445548128905, 0.4524792282339751, 0.4496814515545284, 0.44326407298234644, 0.44824244257494533, 0.43535236913172565, 0.43220757035473345, 0.42938324929099086, 0.42652773593274457, 0.42282234017733933, 0.427682207301929, 0.42007629833667887, 0.42402261607728986, 0.4139103795358925, 0.3738628578112873, 0.40080944896084975, 0.3883825791500647]

        aa = list(range(2, 50))

        print(len(silueta))
        print(len(aa))

        plt.plot(aa, silueta, 'b')
        plt.xlabel('Clústeres')
        plt.ylabel('Puntaje de la silueta')
        plt.title('Metodo de la Silueta')
        plt.show()
        # plt.savefig("Silueta_vs_Clases.png")


if __name__ == '__main__':
    Cluster = clustering()
    Cluster.read_data()
    Cluster.cluster()
