class ParamServer:
    def __init__(self):
        # /home/pocampo/
        # /home/sirui/Documents/Johan_Avila/
        # /home/avila/Documentos/
        self.prefix = "/home/sirui/Documents/Johan_Avila/"

        # pt04_Cloud_clasificado
        # example
        # NIR_clasificado_training
        self.data_file = "example.txt"

        self.dsp_types = ["L", "P", "S", "O", "A", "E", "C"]  # Sum
