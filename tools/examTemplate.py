import os
from tools.tool_functions import get_poject_path

class ExamTemplate():
    def __init__(self,args,cfg):
        args.root_path = args.root_path
        self.args = args
        self.cfg = cfg
        self.root_path = args.root_path
        self.dataset = cfg["DATASET"]
        self.model_type = cfg["MODEL_TYPE"]
        self.prediction_mode = cfg["PREDICTION_MODE"]
        self.data_shape = cfg["DATA_SHAPE"]
        self.ckpt_dir = os.path.join(self.root_path,"checkpoint",str(self.model_type+"_"+self.prediction_mode))
        self.result_dir = os.path.join(self.root_path,"Out_result",str(self.model_type+"_"+self.prediction_mode))


        
