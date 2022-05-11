import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class clfAnalysis:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data(self):
        print("read_data")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/data/training/"
        file += self.parSer.data_file

        data = pd.read_csv(file, sep=" ", header=0)
        self.Classification = np.array(data.Classification)

        self.pcd_xyz = o3d.geometry.PointCloud()
        self.pcd_xyz.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("datos:", len(self.pcd_xyz.points))

#-------------------------------------------------------------------------------------------------------
#Clasificación

#Separación de datos
p_train = 0.8 #Porcentaje de particion
trainc, testc, trc, tec = train_test_split(descriptores,labels, test_size = 1-p_train) #Aleatorio

p_train = 0.6 #Porcentaje de particion
train, prub, tr, pr = train_test_split(descriptores,labels, test_size = 1-p_train) #Aleatorio

p_train = 0.5 #Porcentaje de particion
adjust, test, ad, te = train_test_split(prub,pr, test_size = 1-p_train) #Aleatorio

class clfAnalysis:
    pass

    def RandomForestClassifier(self):

        neigh = RandomForestClassifier(max_depth=15, random_state=0)  #max_depth=numero de arboles #random_state=desde que arbol comnezar
        neigh.fit(train, tr) #train=todos los descriptores=lista{nx8} #tr=datos etiquetados =columna classification
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        a=[]
        i=1;
        while(i<50):
            clf = RandomForestClassifier(max_depth=i, random_state=0)
            clf.fit(train, tr)
            pre=clf.predict(adjust)
            a.append(accuracy_score(ad,pre)*100)
            print(i)
            i=i+1
        plt.plot(a)
        plt.show()

    def KNeighborsClassifier(self):

        neigh = KNeighborsClassifier(n_neighbors=16)
        neigh.fit(train, tr)
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        pre=neigh.predict(test)
        print("Accuracy: ",accuracy_score(te,pre)*100,"%")
        print("F1: ",f1_score(te,pre,average=None)*100,"%")
        a2=accuracy_score(te,pre)*100
        b2=f1_score(te,pre,average=None)*100

    def SVM(self):
        #SVM
        clf = svm.SVC()
        clf.fit(trainc,trc)
        pre=clf.predict(testc)
        print("Accuracy: ",accuracy_score(tec,pre)*100,"%")
        print("F1: ",f1_score(tec,pre,average=None)*100,"%")
        a3=accuracy_score(tec,pre)*100
        b3=f1_score(tec,pre,average=None)*100

    def Gaussiano(self):

        #Gaussiano

        clf = GaussianNB()
        clf.fit(trainc,trc)
        pre=clf.predict(testc)
        print("Accuracy: ",accuracy_score(tec,pre)*100,"%")
        print("F1: ",f1_score(tec,pre,average=None)*100,"%")
        a1=accuracy_score(tec,pre)*100
        b1=f1_score(tec,pre,average=None)*100

    def Rocchio(self):
        
        # Rocchio
        clf = NearestCentroid()
        clf.fit(trainc,trc)
        pre=clf.predict(testc)
        print("Accuracy: ",accuracy_score(tec,pre)*100,"%")
        print("F1: ",f1_score(tec,pre,average=None)*100,"%")
        a2=accuracy_score(tec,pre)*100
        b2=f1_score(tec,pre,average=None)*100

    def DecisionTreeClassifier(self):

        neigh = DecisionTreeClassifier(random_state=40)
        neigh.fit(train, tr)
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        p_train = 0.8 #Porcentaje de particion
        trainc, testc, trc, tec = train_test_split(descriptores,labels, test_size = 1-p_train) #Aleatorio

        p_train = 0.6 #Porcentaje de particion
        train, prub, tr, pr = train_test_split(descriptores,labels, test_size = 1-p_train) #Aleatorio
        p_train = 0.5 #Porcentaje de particion
        adjust, test, ad, te = train_test_split(prub,pr, test_size = 1-p_train) #Aleatorio


    def Ajuste(self):

        #KNN
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
        #16
        """
        neigh = KNeighborsClassifier(n_neighbors=16)
        neigh.fit(train, tr)
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        pre=neigh.predict(test)
        print("Accuracy: ",accuracy_score(te,pre)*100,"%")
        print("F1: ",f1_score(te,pre,average=None)*100,"%")
        a4=accuracy_score(te,pre)*100
        b4=f1_score(te,pre,average=None)*100

        #DecisionTree
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
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        pre=neigh.predict(test)
        print("Accuracy: ",accuracy_score(te,pre)*100,"%")
        print("F1: ",f1_score(te,pre,average=None)*100,"%")
        a5=accuracy_score(te,pre)*100
        b5=f1_score(te,pre,average=None)*100

        #RandomForest
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
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        pre=neigh.predict(test)
        print("Accuracy: ",accuracy_score(te,pre)*100,"%")
        print("F1: ",f1_score(te,pre,average=None)*100,"%")
        a6=accuracy_score(te,pre)*100
        b6=f1_score(te,pre,average=None)*100

        #Adaboost
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
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        pre=neigh.predict(test)
        print("Accuracy: ",accuracy_score(te,pre)*100,"%")
        print("F1: ",f1_score(te,pre,average=None)*100,"%")
        a7=accuracy_score(te,pre)*100
        b7=f1_score(te,pre,average=None)*100

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
        neigh = PassiveAggressiveClassifier(max_iter=2, random_state=0,tol=1e-3)
        neigh.fit(train, tr)
        pre=neigh.predict(adjust)
        print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
        print("F1: ",f1_score(ad,pre,average=None)*100,"%")

        pre=neigh.predict(test)
        print("Accuracy: ",accuracy_score(te,pre)*100,"%")
        print("F1: ",f1_score(te,pre,average=None)*100,"%")
        a8=accuracy_score(te,pre)*100
        b8=f1_score(te,pre,average=None)*100

        # Gaussian Mixture Models

        clf = GaussianMixture(n_components=2, random_state=0).fit(trainc)
        pre=(clf.predict(testc))+1
        print("Accuracy: ",accuracy_score(tec,pre)*100,"%")
        print("F1: ",f1_score(tec,pre,average=None)*100,"%")
        a9=accuracy_score(tec,pre)*100
        b9=f1_score(tec,pre,average=None)*100


        #K-Means

        clf = KMeans(n_clusters=2, random_state=0).fit(trainc)
        pre=(clf.predict(testc))+1
        print("Accuracy: ",accuracy_score(tec,pre)*100,"%")
        print("F1: ",f1_score(tec,pre,average=None)*100,"%")
        a10=accuracy_score(tec,pre)*100
        b10=f1_score(tec,pre,average=None)*100


        #Guardar datos
        import pandas as pd
        from pandas import ExcelWriter
        df = pd.DataFrame({'Clasificador': ['Bayesiano', 'Rochio', 'SVM', 'Knn', 'Arbol Decision', 'Random Forest', 'Adaboost', 'Pasivo agresivo', 'GMM', 'K-Means'],
                        'Accuracy': [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10],
                        'F1_Score_Clase_0': [b1[0],b2[0],b3[0],b4[0],b5[0],b6[0],b7[0],b8[0],b9[0],b10[0]],
                            'F1_Score_Clase_1': [b1[1],b2[1],b3[1],b4[1],b5[1],b6[1],b7[1],b8[1],b9[1],b10[1],]})
        df = df[['Clasificador', 'Accuracy', 'F1_Score_Clase_0','F1_Score_Clase_1']]
        writer = ExcelWriter('C:/Users/juanl/Documents/Universidad - Posgrado/Semestres/Segundo Semestre/Teoría de aprendizaje de Máquinas/proyecto/ejemplo1.xlsx')
        df.to_excel(writer, sheet_name='Completos', index=False)
        writer.save()


#RMSE