from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


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

neigh = DecisionTreeClassifier(random_state=40)
neigh.fit(train, tr)
pre=neigh.predict(adjust)
print("Accuracy: ",accuracy_score(ad,pre)*100,"%")
print("F1: ",f1_score(ad,pre,average=None)*100,"%")

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


p_train = 0.8 #Porcentaje de particion
trainc, testc, trc, tec = train_test_split(descriptores,labels, test_size = 1-p_train) #Aleatorio

p_train = 0.6 #Porcentaje de particion
train, prub, tr, pr = train_test_split(descriptores,labels, test_size = 1-p_train) #Aleatorio
p_train = 0.5 #Porcentaje de particion
adjust, test, ad, te = train_test_split(prub,pr, test_size = 1-p_train) #Aleatorio