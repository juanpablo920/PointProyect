import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster
from params import ParamServer
from scipy.stats import norm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

class clustering:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data_clustering(self):
        print("read_data_clustering")
        file_clustering = ""
        file_clustering += self.parSer.prefix
        file_clustering += "pointProyect/data/validation/"
        file_clustering += self.parSer.data_file_valid_clustering

        data_clustering = pd.read_csv(file_clustering, sep=" ", header=0)
        self.Classification = np.array(data_clustering.Classification)
        
        indices = data_clustering[data_clustering['Classification'] == 2].index
        data_clustering = data_clustering.drop(indices)

        self.pcd_clustering = data_clustering.to_numpy()
        print(self.pcd_clustering)

        print("datos:", self.pcd_clustering.shape)
        print("File: ", file_clustering)

    def read_data(self):
        print("read_data")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/data/validation/"
        file += self.parSer.data_file_valid

        data = pd.read_csv(file, sep=" ", header=0)
        self.Classification = np.array(data.Classification)
        
        indices = data[data['Classification'] == 2].index
        data = data.drop(indices)

        self.pcd = data.to_numpy()
        print(self.pcd)

        print("datos:", self.pcd.shape)
        print("File: ", file)

    def cluster(self):

        print("cluster")
        silueta = []
        for k in range(2, 50):
            print("K: ", k)
            kmeans = KMeans(n_clusters=k).fit(self.pcd_clustering)
            labels = kmeans.labels_
            print(labels)
            silueta.append(silhouette_score(
                self.pcd_clustering, labels, metric='euclidean'))
            print(silueta)
        
        max_silueta = np.max(silueta)
        print(max_silueta)
        self.N_arbol=labels[silueta]
        print(self.N_arbol)

        aa = list(range(2, 50))

        print(len(silueta))
        print(len(aa))

        plt.plot(aa, silueta, 'b')
        plt.xlabel('Clústeres')
        plt.ylabel('Puntaje de la silueta')
        plt.title('Metodo de la Silueta')
        #plt.show()
        plt.savefig("Silueta_vs_Clases.png")

    # def Classification(self):
        
    #     df=pd.read_csv("/home/hmurcia/Downloads/libano.txt",sep=" ")
    #     X=np.zeros([len(df),2])
    #     X[:,0]=df.X.values
    #     X[:,1]=df.Y.values
    #     kmeans = KMeans(n_clusters=self.N_arbol, random_state=0)
    #     kmeans = kmeans.fit(X)
    #     centers=kmeans.cluster_centers_
    #     plt.figure()
    #     plt.plot(X[:,0],X[:,1],'.',label='data')
    #     plt.plot(centers[:,0],centers[:,1],'*',label="Kmenas centers")
    #     plt.legend()
    #     plt.show()

    #     pred = kmeans.predict(X)
    #     plt.figure()
    #     plt.scatter(X[:,0],X[:,1], c = pred)
    #     plt.plot(centers[:,0],centers[:,1],'*',label="Kmenas centers")
    #     plt.legend()
    #     plt.show()


if __name__ == '__main__':
    Cluster = clustering()
    Cluster.read_data()
    Cluster.cluster()
