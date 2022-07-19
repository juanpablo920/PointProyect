import os
import joblib
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time as tm
from params import ParamServer

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import silhouette_score


class clfAnalysis:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data_train(self):
        print("read_data_train")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/data/"

        file_train = file_base + "training/"
        file_train += self.parSer.data_file_train

        # train
        data = pd.read_csv(file_train, sep=" ", header=0)
        self.Classification_train = np.array(data.Classification, dtype=int)
        data = data.drop(['Classification'], axis=1)

        self.pcd_train = o3d.geometry.PointCloud()
        self.pcd_train.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("-> datos_train:", len(self.pcd_train.points))

    def read_data_valid(self):
        print("read_data")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/data/"

        file_valid = file_base + "validation/"
        file_valid += self.parSer.data_file_valid

        # valid
        data = pd.read_csv(file_valid, sep=" ", header=0)
        self.Classification_valid = np.array(data.Classification, dtype=int)
        data = data.drop(['Classification'], axis=1)

        self.pcd_valid = o3d.geometry.PointCloud()
        self.pcd_valid.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("-> datos_valid:", len(self.pcd_valid.points))

    def read_data_dsp_train(self):
        print("read_data_dsp_train")

        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/dsp/"

        file_train = file_base + "dsp_train.txt"

        # train
        data = pd.read_csv(file_train, sep=" ", header=0)
        data = data.drop(['X'], axis=1)
        data = data.drop(['Y'], axis=1)
        data = data.drop(['Z'], axis=1)

        self.Classification_train = np.array(data.Classification, dtype=int)
        data = data.drop(['Classification'], axis=1)

        self.dsp_train = data.to_numpy()

        print("-> Classification_train:", len(self.Classification_train))
        print("-> dsp_train:", self.dsp_train.shape)

    def read_data_dsp_valid(self):
        print("read_data_dsp_valid")

        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/dsp/"

        file_valid = file_base + "dsp_valid.txt"

        # valid
        data = pd.read_csv(file_valid, sep=" ", header=0)
        data = data.drop(['X'], axis=1)
        data = data.drop(['Y'], axis=1)
        data = data.drop(['Z'], axis=1)

        self.Classification_valid = np.array(data.Classification, dtype=int)
        data = data.drop(['Classification'], axis=1)

        self.dsp_valid = data.to_numpy()

        print("-> datos_Classification_valid:", len(self.Classification_valid))
        print("-> dsp_valid:", self.dsp_valid.shape)

    def read_model_clf_type(self, clf_type):
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/models_clf/"

        file = file_base + clf_type + ".pkl"

        clf = joblib.load(file)
        return clf

    def read_results_PCD_validation(self):
        print("read_results_PCD_validation")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/clfAnalysis/data/results_PCD_validation/"
        file += "clf_" + self.parSer.data_file_valid

        data = pd.read_csv(file, sep=" ", header=0)

        indices = data[data['Classification'] == 2].index
        data = data.drop(indices)
        data = data.drop(['Classification'], axis=1)

        self.pcd_results_validation = o3d.geometry.PointCloud()
        self.pcd_results_validation.points = o3d.utility.Vector3dVector(
            data.to_numpy())

        print("File: ", "clf_" + self.parSer.data_file_valid)
        print("datos:", len(self.pcd_results_validation.points))

    def setting_dsp_train(self):
        print("setting_dsp_train")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/dsp/"

        file_train = file_base + "dsp_train.txt"

        encabezado = ""
        encabezado += "X Y Z Classification"
        for dsp_type in self.parSer.dsp_types:
            encabezado += " " + dsp_type

        with open(file_train, 'w') as f:
            f.write(encabezado+"\n")

    def setting_dsp_valid(self):
        print("setting_dsp_valid")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/dsp/"

        file_valid = file_base + "dsp_valid.txt"

        encabezado = ""
        encabezado += "X Y Z Classification"
        for dsp_type in self.parSer.dsp_types:
            encabezado += " " + dsp_type

        with open(file_valid, 'w') as f:
            f.write(encabezado+"\n")

    def setting_report_clf(self):
        print("setting_report_clf")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/models_clf/"
        file = file_base + "report_clf.txt"

        with open(file, 'w') as f:
            f.write("clf accuracy f1_2 f1_16\n")

    def save_dps(self, dsp_type, X, Y, Z, Classification, dsp_values):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/clfAnalysis/data/dsp/"
        file += dsp_type + ".txt"

        linea = ""
        linea += str(X)
        linea += " " + str(Y)
        linea += " " + str(Z)
        linea += " " + str(Classification)

        for dsp_value in dsp_values:
            linea += " " + str(dsp_value)

        with open(file, 'a') as f:
            f.write(linea+"\n")

    def save_model_clf_type(self, clf_type, clf):
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/models_clf/"

        file = file_base + clf_type + ".pkl"

        joblib.dump(clf, file)

    def save_report_clf_type(self, clf, accuracy, f1):
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/models_clf/"
        file = file_base + "report_clf.txt"

        linea = ""
        linea += str(clf)
        linea += " " + str(accuracy)
        linea += " " + str(f1[0])
        linea += " " + str(f1[1])

        with open(file, 'a') as f:
            f.write(linea+"\n")

    def calculo_valores_propios(self, matricesCov):
        val_propio_cov = np.linalg.eigvals(matricesCov)
        val_propio = np.sort(val_propio_cov)
        L1 = val_propio[2]
        L2 = val_propio[1]
        L3 = val_propio[0]
        if(L3 <= 0):
            L3 = 1e-8
        if(L2 <= 0):
            L2 = 1e-8
        if(L1 <= 0):
            L1 = 1e-8
        Sum_L123 = (L1+L2+L3)
        e1 = L1/(Sum_L123)
        e2 = L2/(Sum_L123)
        e3 = L3/(Sum_L123)
        return e1, e2, e3

    def calculo_dsp_type(self, dsp_type, e1, e2, e3):
        dsp_value = 0
        if dsp_type == "L":
            dsp_value = (e1-e2)/(e1)
        elif dsp_type == "P":
            dsp_value = (e2-e3)/(e1)
        elif dsp_type == "S":
            dsp_value = e3/e1
        elif dsp_type == "O":
            dsp_value = np.cbrt(e1*e2*e3)
        elif dsp_type == "A":
            dsp_value = (e1-e3)/(e1)
        elif dsp_type == "E":
            dsp_value = - (e1*np.log(e1) + e2*np.log(e2) +
                           e3*np.log(e3))
        elif dsp_type == "C":
            dsp_value = e3/(e1+e2+e3)
        elif dsp_type == "Sum":
            dsp_value = e1 + e2 + e3
        return dsp_value

    def generate_dsp_train(self, radius):
        print("generate_dsp_train")
        print(">"*10)
        print("-> radius: ", radius)
        print("-> calculo de matrices de covarianza")
        self.pcd_train.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))

        print("-> save_dps_train")
        for idx, matricesCov_tmp in enumerate(self.pcd_train.covariances):
            X, Y, Z = self.pcd_train.points[idx]
            Classification_tmp = self.Classification_train[idx]

            e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)

            dsp_values_tmp = []
            for dsp_type in self.parSer.dsp_types:
                dsp_value = self.calculo_dsp_type(dsp_type, e1, e2, e3)
                dsp_values_tmp.append(dsp_value)

            self.save_dps("dsp_train", X, Y, Z,
                          Classification_tmp, dsp_values_tmp)

    def generate_dsp_valid(self, radius):
        print("generate_dsp_valid")
        print(">"*10)
        print("-> radius: ", radius)
        print("-> calculo de matrices de covarianza")

        self.pcd_valid.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))

        print("-> save_dps_valid")
        for idx, matricesCov_tmp in enumerate(self.pcd_valid.covariances):
            X, Y, Z = self.pcd_valid.points[idx]
            Classification_tmp = self.Classification_valid[idx]

            e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)

            dsp_values_tmp = []
            for dsp_type in self.parSer.dsp_types:
                dsp_value = self.calculo_dsp_type(dsp_type, e1, e2, e3)
                dsp_values_tmp.append(dsp_value)

            self.save_dps("dsp_valid", X, Y, Z,
                          Classification_tmp, dsp_values_tmp)

    def generate_clf_models(self):
        print("generate_dsp_valid")
        print(">"*10)

        print("RandomForest")
        clf = RandomForestClassifier(
            max_depth=len(self.parSer.dsp_types),
            random_state=0)
        clf.fit(self.dsp_train, self.Classification_train)
        self.save_model_clf_type("RandomForest", clf)

        print("KNeighbors")
        clf = KNeighborsClassifier(n_neighbors=len(self.parSer.dsp_types))
        clf.fit(self.dsp_train, self.Classification_train)
        self.save_model_clf_type("KNeighbors", clf)

        # print("SVM")
        # clf = svm.SVC()
        # clf.fit(self.dsp_train, self.Classification_train)
        # self.save_model_clf_type("SVM",clf)

        print("Gaussiano")
        clf = GaussianNB()
        clf.fit(self.dsp_train, self.Classification_train)
        self.save_model_clf_type("Gaussiano", clf)

        print("Rocchio")
        clf = NearestCentroid()
        clf.fit(self.dsp_train, self.Classification_train)
        self.save_model_clf_type("Rocchio", clf)

        print("DecisionTree")
        clf = DecisionTreeClassifier(random_state=len(self.parSer.dsp_types))
        clf.fit(self.dsp_train, self.Classification_train)
        self.save_model_clf_type("DecisionTree", clf)

    def generate_report_clf(self):
        print("generate_report_clf")
        print(">"*10)

        print("RandomForest")
        clf = self.read_model_clf_type("RandomForest")
        pre = clf.predict(self.dsp_valid)

        accuracy = accuracy_score(self.Classification_valid, pre)*100
        f1 = f1_score(self.Classification_valid, pre, average=None)*100

        print("-> Accuracy: ", accuracy, "%")
        print("-> F1: ", f1, "%")

        self.save_report_clf_type("RandomForest", accuracy, f1)

        print("KNeighbors")
        clf = self.read_model_clf_type("KNeighbors")
        pre = clf.predict(self.dsp_valid)

        accuracy = accuracy_score(self.Classification_valid, pre)*100
        f1 = f1_score(self.Classification_valid, pre, average=None)*100

        print("-> Accuracy: ", accuracy, "%")
        print("-> F1: ", f1, "%")

        self.save_report_clf_type("KNeighbors", accuracy, f1)

        # print("SVM")
        # clf = self.read_model_clf_type("SVM")
        # pre = clf.predict(self.dsp_valid)

        # accuracy = accuracy_score(self.Classification_valid, pre)*100
        # f1 = f1_score(self.Classification_valid, pre, average=None)*100

        # print("-> Accuracy: ", accuracy, "%")
        # print("-> F1: ", f1, "%")

        # self.save_report_clf_type("SVM", accuracy, f1)

        print("Gaussiano")
        clf = self.read_model_clf_type("Gaussiano")
        pre = clf.predict(self.dsp_valid)

        accuracy = accuracy_score(self.Classification_valid, pre)*100
        f1 = f1_score(self.Classification_valid, pre, average=None)*100

        print("-> Accuracy: ", accuracy, "%")
        print("-> F1: ", f1, "%")

        self.save_report_clf_type("Gaussiano", accuracy, f1)

        print("Rocchio")
        clf = self.read_model_clf_type("Rocchio")
        pre = clf.predict(self.dsp_valid)

        accuracy = accuracy_score(self.Classification_valid, pre)*100
        f1 = f1_score(self.Classification_valid, pre, average=None)*100

        print("-> Accuracy: ", accuracy, "%")
        print("-> F1: ", f1, "%")

        self.save_report_clf_type("Rocchio", accuracy, f1)

        print("DecisionTree")
        clf = self.read_model_clf_type("DecisionTree")
        pre = clf.predict(self.dsp_valid)

        accuracy = accuracy_score(self.Classification_valid, pre)*100
        f1 = f1_score(self.Classification_valid, pre, average=None)*100

        print("-> Accuracy: ", accuracy, "%")
        print("-> F1: ", f1, "%")

        self.save_report_clf_type("DecisionTree", accuracy, f1)

    def results_PCD_validation(self):
        print("results_PCD_validation")
        print(">"*10)

        print("RandomForest")
        clf = self.read_model_clf_type("RandomForest")
        pre = clf.predict(self.dsp_valid)

        accuracy = accuracy_score(self.Classification_valid, pre)*100
        f1 = f1_score(self.Classification_valid, pre, average=None)*100

        print("-> Accuracy: ", accuracy, "%")
        print("-> F1: ", f1, "%")

        print("save_clf_results")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/clfAnalysis/data/results_PCD_validation/"
        file += "clf_" + self.parSer.data_file_valid

        with open(file, 'w') as f:
            f.write("X Y Z Classification\n")
            for idx, XYZ in enumerate(self.pcd_valid.points):
                X, Y, Z = XYZ
                f.write(str(X)+" "+str(Y)+" "+str(Z) +
                        " "+str(pre[idx])+"\n")

    def individual_tree_segmentation(self, min_num_tree, max_num_tree):
        print("individual_tree_segmentation")
        print(">"*10)
        print("-> lowResolutionPcd")
        lowPcd_xyz = self.pcd_results_validation.uniform_down_sample(10)
        lowPcd_xyz = np.array(lowPcd_xyz.points)
        print("-> low datos:", lowPcd_xyz.shape)

        print("-> cluster")
        indices_silueta = []
        for k in range(min_num_tree, max_num_tree):
            print("-> K: ", k)

            kmeans = KMeans(n_clusters=k).fit(lowPcd_xyz)
            Classification_cluster = kmeans.labels_

            tmp_silueta = silhouette_score(
                lowPcd_xyz,
                Classification_cluster,
                metric='euclidean')

            indices_silueta.append(tmp_silueta)

        max_silueta = np.amax(indices_silueta)
        posMax_silueta = np.where(indices_silueta == max_silueta)

        print(posMax_silueta)
        print(posMax_silueta+2)

        aa = list(range(2, 50))

        print(len(indices_silueta))
        print(len(aa))

        plt.plot(aa, indices_silueta, 'b')
        plt.xlabel('Clústeres')
        plt.ylabel('Puntaje de la silueta')
        plt.title('Metodo de la Silueta')
        # plt.show()
        plt.savefig("Silueta_vs_Clases.png")


if __name__ == '__main__':
    clf_analysis = clfAnalysis()

    print("Opcion_1: generar archivos dps train")
    print("Opcion_2: generar archivos dps valid")
    print("")
    print("Opcion_3: generar modelos clf")
    print("Opcion_4: generar reporte clf")
    print("Opcion_5: generar nube de puntos clasificada")
    print("")
    print("Opcion_6: segmentación individual de árboles")

    opcion = input("opcion: ")

    if opcion == "1":
        print("="*10)
        print("generar archivos dps train")
        print("")
        print("Opcion_1: init_files_dsp_train")
        print("Opcion_x: salir")
        opcion = input("opcion: ")
        if opcion != "1":
            exit()
        print("")
        clf_analysis.setting_dsp_train()
        clf_analysis.read_data_train()
        print("-"*10)
        radius_dsp = float(input("radius: "))
        print("-"*10)
        clf_analysis.generate_dsp_train(radius_dsp)
    elif opcion == "2":
        print("="*10)
        print("generar archivos dps valid")
        print("")
        print("Opcion_1: init_files_dsp_valid")
        print("Opcion_x: salir")
        opcion = input("opcion: ")
        if opcion != "1":
            exit()
        print("")
        clf_analysis.setting_dsp_valid()
        clf_analysis.read_data_valid()
        print("-"*10)
        radius_dsp = float(input("radius: "))
        print("-"*10)
        clf_analysis.generate_dsp_valid(radius_dsp)
    elif opcion == "3":
        print("="*10)
        print("generar modelos clf")
        print("")
        clf_analysis.read_data_dsp_train()
        clf_analysis.generate_clf_models()
    elif opcion == "4":
        print("="*10)
        print("generar reporte clf")
        print("")
        clf_analysis.setting_report_clf()
        clf_analysis.read_data_dsp_valid()
        clf_analysis.generate_report_clf()
    elif opcion == "5":
        print("="*10)
        print("generar nube de puntos clasificada")
        print("")
        clf_analysis.read_data_valid()
        clf_analysis.read_data_dsp_valid()
        clf_analysis.results_PCD_validation()
    elif opcion == "6":
        print("="*10)
        print("segmentación individual de árboles")
        clf_analysis.read_results_PCD_validation()
        print("-"*10)
        print("numero de arboles aprox")
        min_num_tree = float(input("de: "))
        max_num_tree = float(input("a: "))
        print("-"*10)
        clf_analysis.individual_tree_segmentation(min_num_tree, max_num_tree)
        print("")
    else:
        print("="*10)
        print("no es una opcion '{opcion}'")
