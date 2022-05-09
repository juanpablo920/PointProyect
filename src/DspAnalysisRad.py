import os
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time as tm
from params import ParamServer


class dpsAnalysis:

    def __init__(self):
        self.parSer = ParamServer()

    def read_data(self):
        print("read_data")
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/data/training/"
        file += self.parSer.data_file

        data = pd.read_csv(file, sep=" ", header=0)
        # self.X = np.array(data.X)
        # self.Y = np.array(data.Y)
        # self.Z = np.array(data.Z)
        self.Classification = np.array(data.Classification)
        data = data.drop(['Classification'], axis=1)

        self.pcd_xyz = o3d.geometry.PointCloud()
        self.pcd_xyz.points = o3d.utility.Vector3dVector(data.to_numpy())

        print("datos:", len(self.pcd_xyz.points))

    def setting_files_P123(self):
        print("setting_files_P123")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/dpsAnalysis/radius/data/P123/P_"

        for dsp_type in self.parSer.dsp_types:
            file_P123 = file_base + dsp_type + ".txt"
            with open(file_P123, 'w') as f:
                f.write("radius P12 P13 P32\n")

        file_time = file_base + "time.txt"
        with open(file_time, 'w') as f:
            f.write("radius time\n")

    def setting_files_dsp(self):
        print("setting_files_dsp")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/dpsAnalysis/radius/data/dsp/"

        for dsp_type in self.parSer.dsp_types:
            file_dsp = file_base + dsp_type + ".txt"
            with open(file_dsp, 'w') as f:
                f.write("value\n")

        file_time = file_base + "radius.txt"
        with open(file_time, 'w') as f:
            f.write("radius\n")

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

    def save_data_P123_dps_type(self, dsp_type, radius, P12, P13, P32):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis/radius/data/P123/P_"
        file += dsp_type + ".txt"
        with open(file, 'a') as f:
            f.write(
                str(radius)+" " + str(P12)+" " + str(P13)+" "+str(P32)+"\n")

    def save_data_P123_time(self, radius, time):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis/radius/data/P123/P_time.txt"
        with open(file, 'a') as f:
            f.write(str(radius)+" "+str(time)+"\n")

    def save_data_dps_type(self, dsp_type, dsp_value):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis/radius/data/dsp/"
        file += dsp_type + ".txt"
        with open(file, 'a') as f:
            f.write(str(dsp_value)+"\n")

    def save_data_dsp_radius(self, radius):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis/radius/data/dsp/radius.txt"
        with open(file, 'a') as f:
            f.write(str(radius)+"\n")

    def generate_files_P123(self, radius_init, radius_finish, radius_steps):
        print("generate_files_P123")
        for radius in np.arange(radius_init, radius_finish, radius_steps):
            time_inicio = tm.time()
            print(">"*10)
            print("-> radius: ", radius)
            print("-> calculo de matrices de covarianza")
            self.pcd_xyz.estimate_covariances(
                search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))

            print("-> calculo valores propios e")
            e = []
            for matricesCov_tmp in self.pcd_xyz.covariances:
                e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)
                e.append([e1, e2, e3])

            print("-> save_data_P123_dps_type")
            for dsp_type in self.parSer.dsp_types:

                dsp_value_tmp = [[], [], []]
                for idx, e_tmp in enumerate(e):
                    e1, e2, e3 = e_tmp
                    dsp_value = self.calculo_dsp_type(dsp_type, e1, e2, e3)

                    if self.Classification[idx] == 16:  # Tree
                        dsp_value_tmp[0].append(dsp_value)
                    elif self.Classification[idx] == 2:  # ground
                        dsp_value_tmp[1].append(dsp_value)
                    elif self.Classification[idx] == 8:  # Model_keypoints
                        dsp_value_tmp[2].append(dsp_value)

                tree_mean = np.mean(dsp_value_tmp[0])
                tree_std = np.std(dsp_value_tmp[0])
                ground_mean = np.mean(dsp_value_tmp[1])
                ground_std = np.std(dsp_value_tmp[1])
                marcador_mean = np.mean(dsp_value_tmp[2])
                marcador_std = np.std(dsp_value_tmp[2])

                #Arbol - suelo
                P12 = (
                    np.abs(tree_mean - ground_mean)) / (3*(tree_std + ground_std))
                #Arbol - Marcador
                P13 = (
                    np.abs(tree_mean - marcador_mean)) / (3*(tree_std + marcador_std))
                #Marcador - Suelo
                P32 = (
                    np.abs(marcador_mean - ground_mean)) / (3*(marcador_std + ground_std))
                self.save_data_P123_dps_type(dsp_type, radius, P12, P13, P32)

            e = None
            dsp_value_tmp = None
            time = tm.time() - time_inicio
            self.save_data_P123_time(radius, time)

    def generate_files_dsp(self, radius):
        print("generate_files_dsp")
        print(">"*10)
        print("-> radius: ", radius)
        print("-> calculo de matrices de covarianza")
        self.pcd_xyz.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))

        print("-> calculo valores propios e")
        e = []
        for matricesCov_tmp in self.pcd_xyz.covariances:
            e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)
            e.append([e1, e2, e3])

        print("-> save_data_dps_type")
        for dsp_type in self.parSer.dsp_types:
            for e_tmp in e:
                e1, e2, e3 = e_tmp
                dsp_value = self.calculo_dsp_type(dsp_type, e1, e2, e3)
                self.save_data_dps_type(dsp_type, dsp_value)

        self.save_data_dsp_radius(radius)

    def graph_P123_dps_type(self, dps_type, P12, P13, P32, radius):
        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/radius/images/graphics_P123/"

        max_P12 = np.amax(P12)
        max_P13 = np.amax(P13)
        max_P32 = np.amax(P32)

        posMax_P12 = np.where(P12 == max_P12)
        posMax_P13 = np.where(P13 == max_P13)
        posMax_P32 = np.where(P32 == max_P32)

        plt.figure()
        plt.plot(radius, P12, 'C0', label="Arbol_suelo")
        plt.plot(
            radius[posMax_P12], P12[posMax_P12],
            'vC0', label="P(max): {0:.2f}".format(max_P12))

        plt.plot(radius, P13, 'C1', label="Arbol_Marcador")
        plt.plot(
            radius[posMax_P13], P13[posMax_P13],
            'vC1', label="P(max): {0:.2f}".format(max_P13))

        plt.plot(radius, P32, 'C2', label="Marcador_Suelo")
        plt.plot(
            radius[posMax_P32], P32[posMax_P32],
            'vC2', label="P(max): {0:.2f}".format(max_P32))

        plt.title(dps_type + "_vs_radius")
        plt.xlabel('radius')
        plt.ylabel('P')
        plt.grid(True)
        plt.legend()
        plt.savefig(pwd_imagen + dps_type + "_vs_radius.png")
        plt.clf()

    def graph_time_P(self, time, radius):
        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/radius/images/graphics_P123/"

        plt.figure()
        plt.plot(radius, time, "C0", label="time")
        plt.title("time_vs_radius")
        plt.xlabel('radius')
        plt.ylabel('time')
        plt.grid(True)
        plt.legend()
        plt.savefig(pwd_imagen + "time_vs_radius.png")
        plt.clf()

    def graph_average_P123_dps_type(self, dps_type, P12, P13, P32, radius):
        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/radius/images/average_P123/"

        average_P = (P12 + P13 + P32)/3
        max_average = np.amax(average_P)
        posMax_average = np.where(average_P == max_average)

        plt.figure()
        plt.plot(radius, average_P, "C0", label="average_P")
        plt.plot(
            radius[posMax_average], average_P[posMax_average],
            'vC0', label="average(max): {0:.2f} \n radius: {1:.2f}".format(max_average, radius[posMax_average][0]))
        plt.title("average_"+dps_type+"_vs_radius")
        plt.xlabel('radius')
        plt.ylabel('average_P')
        plt.grid(True)
        plt.legend()
        plt.savefig(pwd_imagen + "average_"+dps_type+"_vs_radius.png")
        plt.clf()

        return average_P

    def graph_selection_Radius(self, averages_P, radius):
        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/radius/images/selection_Radius/"

        max_averages_P = np.amax(averages_P)
        posMax_averages = np.where(averages_P == max_averages_P)

        plt.figure()
        plt.plot(radius, averages_P, "C0", label="averages_P")
        plt.plot(
            radius[posMax_averages], averages_P[posMax_averages],
            'vC0', label="average(max): {0:.2f} \n radius: {1:.2f}".format(max_averages_P, radius[posMax_averages][0]))
        plt.title("averages_P_vs_radius")
        plt.xlabel('radius')
        plt.ylabel('averages_P')
        plt.grid(True)
        plt.legend()
        plt.savefig(pwd_imagen + "averages_P_vs_radius.png")
        plt.clf()

    def generate_graphics_P123(self):
        pwd_files = ""
        pwd_files += self.parSer.prefix
        pwd_files += "pointProyect/dpsAnalysis/radius/data/P123/"

        name_files = os.listdir(pwd_files)
        name_files.remove("data.txt")
        name_files.remove("P_time.txt")

        list_averages_P = []
        print("")
        for name_file in name_files:
            data = pd.read_csv(pwd_files+name_file, sep=" ", header=0)
            print("--> ", name_file)

            radius = np.array(data.radius)

            #Arbol - suelo
            P12 = np.array(data.P12)
            #Arbol - Marcador
            P13 = np.array(data.P13)
            #Marcador - Suelo
            P32 = np.array(data.P32)

            dps_type = name_file[:len(name_file)-4]
            self.graph_P123_dps_type(dps_type, P12, P13, P32, radius)
            average_P_tmp = self.graph_average_P123_dps_type(
                dps_type, P12, P13, P32, radius)

            list_averages_P.append(average_P_tmp)

        times = pd.read_csv(pwd_files+"P_time.txt", sep=" ", header=0)
        print("-->  P_time.txt")

        radius = np.array(times.radius)
        time = np.array(times.time)
        self.graph_time_P(time, radius)

        averages_P = np.sum(list_averages_P, axis=0)/len(list_averages_P)
        self.graph_selection_Radius(averages_P, radius)

    def graph_gaussiana_dps_type(self, dps_type, dsp_tree, dsp_ground, dsp_marcador):
        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/radius/images/gaussiana_dps/"

        tree_mean = np.mean(dsp_tree)
        tree_std = np.std(dsp_tree)

        ground_mean = np.mean(dsp_ground)
        ground_std = np.std(dsp_ground)

        marcador_mean = np.mean(dsp_marcador)
        marcador_std = np.std(dsp_marcador)

        Clase1dls = norm.pdf(dsp_tree, tree_mean, tree_std)
        Clase2dls = norm.pdf(dsp_ground, ground_mean, ground_std)
        Clase3dls = norm.pdf(dsp_marcador, marcador_mean, marcador_std)

        plt.figure()
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.plot(dsp_tree, Clase1dls, 'b', label="Arbol")
        plt.plot(dsp_ground, Clase2dls, 'r', label="Tierra")
        plt.plot(dsp_marcador, Clase3dls, 'g', label="Marcador")
        plt.title("Distribucion normal " + dps_type)
        plt.legend()
        plt.savefig(pwd_imagen + "gaussiana_" + dps_type + ".png")
        plt.clf()

    def graph_histograms_dps(self, dps_type, dsp_values):
        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/radius/images/histograms_dps/"

        plt.figure()
        plt.hist(dsp_values, bins=255, label=dps_type)
        plt.title("Histograma " + dps_type)
        plt.legend()
        plt.savefig(pwd_imagen + "histograms_" + dps_type + ".png")
        plt.clf()

    def generate_graphics_dsp(self):
        pwd_files = ""
        pwd_files += self.parSer.prefix
        pwd_files += "pointProyect/dpsAnalysis/radius/data/dsp/"

        name_files = os.listdir(pwd_files)
        name_files.remove("data.txt")
        name_files.remove("radius.txt")

        pos_Tree = self.Classification == 16  # Tree
        pos_ground = self.Classification == 2  # ground
        pos_Model_keypoints = self.Classification == 8  # Model_keypoints

        print("")
        for name_file in name_files:
            data = pd.read_csv(pwd_files+name_file, sep=" ", header=0)
            print("--> ", name_file)

            dsp_values = np.array(data.value)

            dsp_tree = np.sort(dsp_values[pos_Tree])
            dsp_ground = np.sort(dsp_values[pos_ground])
            dsp_marcador = np.sort(dsp_values[pos_Model_keypoints])

            dps_type = name_file[:len(name_file)-4]
            self.graph_gaussiana_dps_type(
                dps_type, dsp_tree, dsp_ground, dsp_marcador)

            self.graph_histograms_dps(dps_type, dsp_values)


if __name__ == '__main__':
    dps_analysis = dpsAnalysis()

    print("Opcion_1: generar archivos P123")
    print("Opcion_2: generar graficas P123")
    print("")
    print("Opcion_3: generar archivos dps")
    print("Opcion_4: generar graficas dps")
    print("")

    opcion = input("opcion: ")

    if opcion == "1":
        print("="*10)
        print("generar archivos P123")
        print("")
        print("Opcion_1: init_files_P123")
        print("Opcion_x: continue")
        opcion = input("opcion: ")
        if opcion == "1":
            dps_analysis.setting_files_P123()
        print("")
        dps_analysis.read_data()
        print("-"*10)
        radius_init = float(input("radius_init: "))
        radius_finish = float(input("radius_finish: "))
        radius_steps = float(input("radius_steps: "))
        print("-"*10)
        dps_analysis.generate_files_P123(
            radius_init, radius_finish, radius_steps)
    elif opcion == "2":
        print("="*10)
        print("generar graficas P123")
        dps_analysis.generate_graphics_P123()
        print("")
    elif opcion == "3":
        print("="*10)
        print("generar archivos dsp")
        print("")
        print("Opcion_1: continue")
        print("Opcion_x: salir")
        opcion = input("opcion: ")
        if opcion != "1":
            exit()
        print("")
        dps_analysis.setting_files_dsp()
        dps_analysis.read_data()
        print("-"*10)
        radius_dsp = float(input("radius: "))
        print("-"*10)
        dps_analysis.generate_files_dsp(radius_dsp)
    elif opcion == "4":
        print("="*10)
        print("generar graficas dsp")
        dps_analysis.read_data()
        dps_analysis.generate_graphics_dsp()
        print("")
    else:
        print("="*10)
        print("no es una opcion '{opcion}'")
