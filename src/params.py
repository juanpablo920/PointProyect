class ParamServer:
    def __init__(self):
        # /home/pocampo/
        # /home/sirui/Documents/Johan_Avila/
        # /home/avila/Documentos/
        # /home/juanpablo
        self.prefix = "/home/sirui/Documents/Johan_Avila/"

        # example
        self.data_file = "PCD_NIR_training_4M_low10.txt"
        self.dsp_types = ["L", "P", "S", "O", "A", "E", "C"]  # Sum
        #self.dsp_types = ["P"]
