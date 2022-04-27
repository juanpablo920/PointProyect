class ParamServer:
    def __init__(self):
        # /home/pocampo/
        # /home/sirui/Documents/Johan_Avila/
        # /home/avila/Documentos/
        # /home/juanocampo/Documentos/Trabajo_de_grado_PointProyect/
        self.prefix = "/home/pocampo/"

        # example
        # PCD_NIR_training_4M_low10
        self.data_file = "PCD_NIR_training_4M_low33.txt"

        self.dsp_types = ["L", "P", "S", "O", "A", "E", "C"]  # Sum
