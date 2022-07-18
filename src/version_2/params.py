class ParamServer:
    def __init__(self):
        # /home/pocampo/
        # /home/sirui/pi_ws/src/
        # /home/avila/Documentos/
        # /home/juanpablo/
        self.prefix = "/home/pocampo/"

        # example_train
        # PCD_NIR_training_4M_low50

        self.dsp_types = ["L", "P", "S", "O", "A", "E", "C"]  # Sum

        self.data_file_train = "coffee_citricos_training_34arboles.txt"
        self.data_file_valid = "Nataima_citricos_etiquetado_validation_15arboles.txt"

        self.clf_P_train = 0.7  # Porcentaje de particion
        self.clf_P_adjust = (1-self.clf_P_train)/(2)
        self.clf_P_validation = self.clf_P_adjust
