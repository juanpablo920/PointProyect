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

        self.data_file_train = "PCD_NIR_training_4M_arbol_suelo.txt"
        self.data_file_valid = "PCD_NIR_validation_4M_arbol_suelo.txt"
