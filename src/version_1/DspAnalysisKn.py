import os
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
from version_2.params import ParamServer


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
        self.Classification = np.array(data.Classification)
        data = data.drop(['Classification'], axis=1)

        self.pcd_xyz = o3d.geometry.PointCloud()
        self.pcd_xyz.points = o3d.utility.Vector3dVector(data.to_numpy())

        pos_Tree = self.Classification == 16  # Tree
        pos_ground = self.Classification == 2  # ground
        pos_Model_keypoints = self.Classification == 8  # Model_keypoints

        colors = np.zeros((data.shape[0], 3))
        colors[pos_Tree, :] = [0.136, 0.556, 0.136]
        colors[pos_ground, :] = [0.512, 0.256, 0]
        colors[pos_Model_keypoints, :] = [0.624, 0.624, 0.624]

        self.pcd_xyz.colors = o3d.utility.Vector3dVector(colors)
        print("datos:", len(self.pcd_xyz.points))

    def init_files_P(self):
        print("setting_files_P")
        file_base = ""
        file_base += self.parSer.prefix
        file_base += "pointProyect/dpsAnalysis/kn/data/P_"
        for dsp_type in self.parSer.dsp_types:
            file_tmp = file_base + dsp_type + ".txt"
            with open(file_tmp, 'w') as f:
                f.write("kn P12 P13 P32\n")

        file_time = file_base + "time.txt"
        with open(file_time, 'w') as f:
            f.write("kn time\n")

    def calculo_valores_propios(self, matricesCov):
        val_propio_cov = np.linalg.eigvals(matricesCov)
        val_propio = np.sort(val_propio_cov)
        L1 = val_propio[2]
        L2 = val_propio[1]
        L3 = val_propio[0]
        Sum_L123 = (L1+L2+L3)
        e1 = L1/(Sum_L123)
        e2 = L2/(Sum_L123)
        e3 = L3/(Sum_L123)
        return e1, e2, e3

    def calculo_dsp(self, dsp_type, e1, e2, e3):
        dsp_value = 0
        if(e1 > 0 and e2 > 0 and e3 > 0):
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
            elif dsp_type == "Sum":
                dsp_value = e1 + e2 + e3
            elif dsp_type == "C":
                dsp_value = e3/(e1+e2+e3)
        return dsp_value

    def save_data_P_dps_type(self, dsp_type, kn, P12, P13, P32):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis/kn/data/P_"
        file += dsp_type + ".txt"
        with open(file, 'a') as f:
            f.write(
                str(kn)+" " + str(P12)+" " + str(P13)+" "+str(P32)+"\n")

    def save_data_P_time(self, kn, time):
        file = ""
        file += self.parSer.prefix
        file += "pointProyect/dpsAnalysis/kn/data/P_time.txt"
        with open(file, 'a') as f:
            f.write(str(kn)+" " + str(time)+"\n")

    def knn(self, kn_init, kn_finish, kn_steps):
        print("knn")
        for kn in range(kn_init, kn_finish, kn_steps):
            time_inicio = tm.time()
            print(">"*10)
            print("-> kn: ", kn)
            print("-> calculo de matrices de covarianza")
            self.pcd_xyz.estimate_covariances(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=kn))

            print("-> calculo valores propios e")
            e = []
            for matricesCov_tmp in self.pcd_xyz.covariances:
                e1, e2, e3 = self.calculo_valores_propios(matricesCov_tmp)
                e.append([e1, e2, e3])

            print("-> dsp_types -> save data P")
            for dsp_type in self.parSer.dsp_types:
                dsp_value_tmp = [[], [], []]
                for idx, e_tmp in enumerate(e):
                    e1, e2, e3 = e_tmp
                    if(e1 < 0 or e2 < 0 or e3 < 0):
                        continue
                    dsp_value = self.calculo_dsp(dsp_type, e1, e2, e3)
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

                self.save_data_P_dps_type(dsp_type, kn, P12, P13, P32)
            e = None
            dsp_value_tmp = None
            time = tm.time() - time_inicio
            self.save_data_P_time(kn, time)

    def view_grafico_gauss(self):
        print("view_grafico_gauss")
        plt.rcParams['agg.path.chunksize'] = 10000

        pwd_imagen = ""
        pwd_imagen += self.parSer.prefix
        pwd_imagen += "pointProyect/dpsAnalysis/kn/images/"

        pwd_files = ""
        pwd_files += self.parSer.prefix
        pwd_files += "pointProyect/dpsAnalysis/kn/data/"

        name_files = os.listdir(pwd_files)
        name_files.remove("data.txt")
        name_files.remove("P_time.txt")

        for name_file in name_files:
            data = pd.read_csv(pwd_files+name_file, sep=" ", header=0)
            print("-"*10)
            print("name_file: ", name_file)
            print(data)

            kn = np.array(data.kn)

            #Arbol - suelo
            P12 = np.array(data.P12)
            #Arbol - Marcador
            P13 = np.array(data.P13)
            #Marcador - Suelo
            P32 = np.array(data.P32)

            max_P12 = np.amax(P12)
            max_P13 = np.amax(P13)
            max_P32 = np.amax(P32)

            posMax_P12 = np.where(P12 == max_P12)
            posMax_P13 = np.where(P13 == max_P13)
            posMax_P32 = np.where(P32 == max_P32)

            name_x = name_file[:len(name_file)-4]

            plt.figure()
            plt.plot(kn, P12, 'C0', label="Arbol_suelo")
            plt.plot(
                kn[posMax_P12], P12[posMax_P12],
                'vC0', label="P(max): {0:.2f}".format(max_P12))

            plt.plot(kn, P13, 'C1', label="Arbol_Marcador")
            plt.plot(
                kn[posMax_P13], P13[posMax_P13],
                'vC1', label="P(max): {0:.2f}".format(max_P13))

            plt.plot(kn, P32, 'C2', label="Marcador_Suelo")
            plt.plot(
                kn[posMax_P32], P32[posMax_P32],
                'vC2', label="P(max): {0:.2f}".format(max_P32))

            plt.title(name_x + "_vs_kn")
            plt.xlabel('kn')
            plt.ylabel('P')
            plt.grid(True)
            plt.legend()
            plt.savefig(pwd_imagen + name_x + "_vs_kn.png")
            plt.clf()

        times = pd.read_csv(pwd_files+"P_time.txt", sep=" ", header=0)
        print("-"*10)
        print("name_file: ", "P_time.txt")
        print(times)

        kn = np.array(times.kn)
        time = np.array(times.time)

        plt.figure()
        plt.plot(kn, time, "C0", label="time")
        plt.title("time_vs_kn")
        plt.xlabel('kn')
        plt.ylabel('time')
        plt.grid(True)
        plt.legend()
        plt.savefig(pwd_imagen + "time_vs_kn.png")
        plt.clf()


if __name__ == '__main__':
    dps_analysis = dpsAnalysis()

    print("Opcion_1: generar archivos P")
    print("Opcion_2: graficas P")

    opcion = input("opcion: ")

    if opcion == "1":
        print("="*10)
        print("generar archivos P")
        print("-"*10)
        dps_analysis.read_data()
        print("-"*10)
        print("Opcion_1: init_files_P")
        print("Opcion_x: continue")
        opcion = input("opcion: ")
        if opcion == "1":
            dps_analysis.init_files_P()
        print("-"*10)
        kn_init = int(input("kn_init: "))
        kn_finish = int(input("kn_finish: "))
        kn_steps = int(input("kn_steps: "))
        print("-"*10)
        dps_analysis.knn(kn_init, kn_finish, kn_steps)
    elif opcion == "2":
        print("="*10)
        print(" graficas P")
        dps_analysis.view_grafico_gauss()
    else:
        print("="*10)
        print("no es una opción '{opcion}'")
