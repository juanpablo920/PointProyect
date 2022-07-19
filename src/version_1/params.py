class ParamServer:
    def __init__(self):
        # /home/pocampo/
        # /home/sirui/pi_ws/src/
        # /home/avila/Documentos/
        # /home/juanpablo/
        self.prefix = "/home/juanpablo/"

        # example_train
        # PCD_NIR_training_4M_low50

        self.dsp_types = ["L", "P", "S", "O", "A", "E", "C"]  # Sum

        self.data_file_train = "Libano_coffee_etiquetado_training_34arboles_low_10.txt"
        self.data_file_valid = "Libano_coffee_etiquetado_validation_15arboles_low_10.txt"

        self.clf_P_train = 0.7  # Porcentaje de particion
        self.clf_P_adjust = (1-self.clf_P_train)/(2)
        self.clf_P_validation = self.clf_P_adjust
