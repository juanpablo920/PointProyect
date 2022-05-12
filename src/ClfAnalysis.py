import os
from unittest.mock import seal
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


class clfAnalysis:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data(self):
        print("read_data")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/data/"

        file_train = file_base + "training/"
        file_train += self.parSer.data_file_train

        file_valid = file_base + "validation/"
        file_valid += self.parSer.data_file_valid

        # train
        data = pd.read_csv(file_train, sep=" ", header=0)
        self.Classification_train = np.array(data.Classification, dtype=int)
        data = data.drop(['Classification'], axis=1)

        self.pcd_train = o3d.geometry.PointCloud()
        self.pcd_train.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("-> datos_train:", len(self.pcd_train.points))

        # valid
        data = pd.read_csv(file_valid, sep=" ", header=0)
        self.Classification_valid = np.array(data.Classification, dtype=int)
        data = data.drop(['Classification'], axis=1)

        self.pcd_valid = o3d.geometry.PointCloud()
        self.pcd_valid.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("-> datos_valid:", len(self.pcd_valid.points))

    def read_data_dsp(self):
        print("read_data_dsp")

        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/"

        file_train = file_base + "dsp_train.txt"
        file_valid = file_base + "dsp_valid.txt"

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

    def setting_files_dsp(self):
        print("setting_files_dsp")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/clfAnalysis/data/"

        file_train = file_base + "dsp_train.txt"
        file_valid = file_base + "dsp_valid.txt"

        encabezado = ""
        encabezado += "X Y Z Classification"
        for dsp_type in self.parSer.dsp_types:
            encabezado += " " + dsp_type

        with open(file_train, 'w') as f:
            f.write(encabezado+"\n")

        with open(file_valid, 'w') as f:
            f.write(encabezado+"\n")

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

    def save_data_dps(self, dsp_type, X, Y, Z, Classification, dsp_values):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/clfAnalysis/data/"
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

    def generate_files_dsp(self, radius):
        print("generate_files_dsp")
        print(">"*10)
        print("-> radius: ", radius)
        print("-> calculo de matrices de covarianza")
        self.pcd_train.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))

        self.pcd_valid.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))

        print("-> save_data_dps_train")
        for idx, matricesCov_tmp in enumerate(self.pcd_train.covariances):
            X, Y, Z = self.pcd_train.points[idx]
            Classification_tmp = self.Classification_train[idx]

            e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)

            dsp_values_tmp = []
            for dsp_type in self.parSer.dsp_types:
                dsp_value = self.calculo_dsp_type(dsp_type, e1, e2, e3)
                dsp_values_tmp.append(dsp_value)

            self.save_data_dps("dsp_train", X, Y, Z,
                               Classification_tmp, dsp_values_tmp)

        print("-> save_data_dps_valid")
        for idx, matricesCov_tmp in enumerate(self.pcd_valid.covariances):
            X, Y, Z = self.pcd_valid.points[idx]
            Classification_tmp = self.Classification_valid[idx]

            e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)

            dsp_values_tmp = []
            for dsp_type in self.parSer.dsp_types:
                dsp_value = self.calculo_dsp_type(dsp_type, e1, e2, e3)
                dsp_values_tmp.append(dsp_value)

            self.save_data_dps("dsp_valid", X, Y, Z,
                               Classification_tmp, dsp_values_tmp)

    def RandomForestClassifier(self):

        # max_depth=numero de arboles #random_state=desde que arbol comnezar
        neigh = RandomForestClassifier(max_depth=15, random_state=0)
        # train=todos los descriptores=lista{nx8} #tr=datos etiquetados =columna classification
        neigh.fit(self.dsp, self.Classification)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        a = []
        i = 1
        while(i < 50):
            clf = RandomForestClassifier(max_depth=i, random_state=0)
            clf.fit(train, tr)
            pre = clf.predict(adjust)
            a.append(accuracy_score(ad, pre)*100)
            print(i)
            i = i+1
        plt.plot(a)
        plt.show()

    def KNeighborsClassifier(self):

        neigh = KNeighborsClassifier(n_neighbors=16)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        pre = neigh.predict(test)
        print("Accuracy: ", accuracy_score(te, pre)*100, "%")
        print("F1: ", f1_score(te, pre, average=None)*100, "%")
        a2 = accuracy_score(te, pre)*100
        b2 = f1_score(te, pre, average=None)*100

    def SVM(self):
        print("SVM")
        clf = svm.SVC()
        clf.fit(self.dsp_train, self.Classification_train)
        pre = clf.predict(self.dsp_valid)
        print("-> Accuracy: ",
              accuracy_score(self.Classification_valid, pre)*100, "%")
        print("-> F1: ",
              f1_score(self.Classification_valid, pre, average=None)*100, "%")

    def Gaussiano(self):
        print("Gaussiano")
        clf = GaussianNB()
        clf.fit(self.dsp_train, self.Classification_train)
        pre = clf.predict(self.dsp_valid)
        print("-> Accuracy: ",
              accuracy_score(self.Classification_valid, pre)*100, "%")
        print("-> F1: ",
              f1_score(self.Classification_valid, pre, average=None)*100, "%")

    def Rocchio(self):
        print("Rocchio")
        clf = NearestCentroid()
        clf.fit(self.dsp_train, self.Classification_train)
        pre = clf.predict(self.dsp_valid)
        print("-> Accuracy: ",
              accuracy_score(self.Classification_valid, pre)*100, "%")
        print("-> F1: ",
              f1_score(self.Classification_valid, pre, average=None)*100, "%")

    def DecisionTreeClassifier(self):

        neigh = DecisionTreeClassifier(random_state=40)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        p_train = 0.8  # Porcentaje de particion
        trainc, testc, trc, tec = train_test_split(
            descriptores, labels, test_size=1-p_train)  # Aleatorio

        p_train = 0.6  # Porcentaje de particion
        train, prub, tr, pr = train_test_split(
            descriptores, labels, test_size=1-p_train)  # Aleatorio
        p_train = 0.5  # Porcentaje de particion
        adjust, test, ad, te = train_test_split(
            prub, pr, test_size=1-p_train)  # Aleatorio

    def Ajuste(self):

        # KNN
        # 300 se ve la caida
        # 10
        """
        a=[]
        i=40;
        while(i<100):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(train, tr)
            pre=neigh.predict(adjust)
            a.append(accuracy_score(ad,pre)*100)
            i=i+1
            print(i)
        plt.plot(a)
        plt.show()
        # 16
        """
        neigh = KNeighborsClassifier(n_neighbors=16)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        pre = neigh.predict(test)
        print("Accuracy: ", accuracy_score(te, pre)*100, "%")
        print("F1: ", f1_score(te, pre, average=None)*100, "%")
        a4 = accuracy_score(te, pre)*100
        b4 = f1_score(te, pre, average=None)*100

        # DecisionTree
        """
        a=[]
        i=1;
        while(i<50):
            clf = DecisionTreeClassifier(random_state=i)
            clf.fit(train, tr)
            pre=clf.predict(adjust)
            a.append(accuracy_score(ad,pre)*100)
            i=i+1
        plt.plot(a)
        plt.show()
        """
        neigh = DecisionTreeClassifier(random_state=40)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        pre = neigh.predict(test)
        print("Accuracy: ", accuracy_score(te, pre)*100, "%")
        print("F1: ", f1_score(te, pre, average=None)*100, "%")
        a5 = accuracy_score(te, pre)*100
        b5 = f1_score(te, pre, average=None)*100

        # RandomForest
        """
        a=[]
        i=1;
        while(i<50):
            clf = RandomForestClassifier(max_depth=i, random_state=0)
            clf.fit(train, tr)
            pre=clf.predict(adjust)
            a.append(accuracy_score(ad,pre)*100)
            i=i+1
        plt.plot(a)
        plt.show()
        """
        neigh = RandomForestClassifier(max_depth=21, random_state=0)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        pre = neigh.predict(test)
        print("Accuracy: ", accuracy_score(te, pre)*100, "%")
        print("F1: ", f1_score(te, pre, average=None)*100, "%")
        a6 = accuracy_score(te, pre)*100
        b6 = f1_score(te, pre, average=None)*100

        # Adaboost
        """
        a=[]
        i=1;
        while(i<50):
            clf = AdaBoostClassifier(n_estimators=i, random_state=0)
            clf.fit(train, tr)
            pre=clf.predict(adjust)
            a.append(accuracy_score(ad,pre)*100)
            i=i+1
        plt.plot(a)
        plt.show()
        """
        neigh = AdaBoostClassifier(n_estimators=35, random_state=0)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        pre = neigh.predict(test)
        print("Accuracy: ", accuracy_score(te, pre)*100, "%")
        print("F1: ", f1_score(te, pre, average=None)*100, "%")
        a7 = accuracy_score(te, pre)*100
        b7 = f1_score(te, pre, average=None)*100

        # Passive Aggressive
        """
        a=[]
        i=1;
        while(i<1001):
            clf = PassiveAggressiveClassifier(max_iter=i, random_state=0,tol=1e-3)
            clf.fit(train, tr)
            pre=clf.predict(adjust)
            a.append(accuracy_score(ad,pre)*100)
            print(i)
            i=i+1
        plt.plot(a)
        plt.show()
        """
        neigh = PassiveAggressiveClassifier(
            max_iter=2, random_state=0, tol=1e-3)
        neigh.fit(train, tr)
        pre = neigh.predict(adjust)
        print("Accuracy: ", accuracy_score(ad, pre)*100, "%")
        print("F1: ", f1_score(ad, pre, average=None)*100, "%")

        pre = neigh.predict(test)
        print("Accuracy: ", accuracy_score(te, pre)*100, "%")
        print("F1: ", f1_score(te, pre, average=None)*100, "%")
        a8 = accuracy_score(te, pre)*100
        b8 = f1_score(te, pre, average=None)*100

        # Gaussian Mixture Models

        clf = GaussianMixture(n_components=2, random_state=0).fit(trainc)
        pre = (clf.predict(testc))+1
        print("Accuracy: ", accuracy_score(tec, pre)*100, "%")
        print("F1: ", f1_score(tec, pre, average=None)*100, "%")
        a9 = accuracy_score(tec, pre)*100
        b9 = f1_score(tec, pre, average=None)*100

        # K-Means

        clf = KMeans(n_clusters=2, random_state=0).fit(trainc)
        pre = (clf.predict(testc))+1
        print("Accuracy: ", accuracy_score(tec, pre)*100, "%")
        print("F1: ", f1_score(tec, pre, average=None)*100, "%")
        a10 = accuracy_score(tec, pre)*100
        b10 = f1_score(tec, pre, average=None)*100

        # Guardar datos
        import pandas as pd
        from pandas import ExcelWriter
        df = pd.DataFrame({'Clasificador': ['Bayesiano', 'Rochio', 'SVM', 'Knn', 'Arbol Decision', 'Random Forest', 'Adaboost', 'Pasivo agresivo', 'GMM', 'K-Means'],
                           'Accuracy': [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10],
                           'F1_Score_Clase_0': [b1[0], b2[0], b3[0], b4[0], b5[0], b6[0], b7[0], b8[0], b9[0], b10[0]],
                           'F1_Score_Clase_1': [b1[1], b2[1], b3[1], b4[1], b5[1], b6[1], b7[1], b8[1], b9[1], b10[1], ]})
        df = df[['Clasificador', 'Accuracy',
                 'F1_Score_Clase_0', 'F1_Score_Clase_1']]
        writer = ExcelWriter(
            'C:/Users/juanl/Documents/Universidad - Posgrado/Semestres/Segundo Semestre/Teoría de aprendizaje de Máquinas/proyecto/ejemplo1.xlsx')
        df.to_excel(writer, sheet_name='Completos', index=False)
        writer.save()


if __name__ == '__main__':
    clf_analysis = clfAnalysis()

    print("Opcion_1: generar archivos dps")
    print("")
    print("Opcion_2: generar archivos clf")
    print("Opcion_3: generar graficas clf")
    print("")

    opcion = input("opcion: ")

    if opcion == "1":
        print("="*10)
        print("generar archivos dsp")
        print("")
        print("Opcion_1: init_files_dsp")
        print("Opcion_x: salir")
        opcion = input("opcion: ")
        if opcion != "1":
            exit()
        print("")
        clf_analysis.setting_files_dsp()
        clf_analysis.read_data()
        print("-"*10)
        radius_dsp = float(input("radius: "))
        print("-"*10)
        clf_analysis.generate_files_dsp(radius_dsp)
    elif opcion == "2":
        print("="*10)
        print("generar archivos clf")
        print("")
        clf_analysis.read_data_dsp()
        print("-"*10)
        clf_analysis.Gaussiano()
        print("")
        clf_analysis.Rocchio()
        print("")
        clf_analysis.SVM()
    elif opcion == "3":
        pass
    else:
        print("="*10)
        print("no es una opcion '{opcion}'")
