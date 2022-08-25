import os
from unittest import result
from tools.examTemplate import ExamTemplate


class InstanceExam(ExamTemplate):
    def __init__(self, args, cfg, exam_cfg, **kwargs):
        super().__init__(args=args, cfg=cfg)
        self.exam_cfg = exam_cfg
        self.compare_mode = exam_cfg["COMPARE_MODE"]
        self.model_method = exam_cfg['MODEL'] if 'MODEL' in exam_cfg else args.modelMethod
        self.input_mode = exam_cfg['INPUT_MODE'] if 'INPUT_MODE' in exam_cfg else args.inputMode
        self.lossFunction_method = exam_cfg['LOSSFUNCTION'] if 'LOSSFUNCTION' in exam_cfg else args.lossFunctionMethod
        self.preImg_num = exam_cfg["PREIMG_NUM"] if "PREIMG_NUM" in exam_cfg else 1
        self.work_fileName = self.methodsName_combine()
        self.inChannel_num = self.get_channelNum()
        self.log_dir, self.csv_dir, self.tensorboard_dir, self.split_img_dir, self.cur_ckpt_dir = self.init_dir()

    def init_dir(self, ):
        log_dir = os.path.join(self.result_dir, self.compare_mode, "log")
        csv_dir = os.path.join(self.result_dir, self.compare_mode, 'csv')
        tensorboard_dir = os.path.join(self.result_dir, self.compare_mode, "run", self.work_fileName)
        cur_ckpt_dir = os.path.join(self.ckpt_dir, self.compare_mode, self.work_fileName)
        os.makedirs(cur_ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        split_img_dir = None
        if self.prediction_mode == "CBCT":
            split_img_dir = os.path.join(self.result_dir, self.compare_mode, 'split_img', self.work_fileName)
            os.makedirs(split_img_dir, exist_ok=True)
        return log_dir, csv_dir, tensorboard_dir, split_img_dir, cur_ckpt_dir

    def get_channelNum(self):
        num = 0
        if self.input_mode.find("origin") != -1:
            num += 1
        if self.input_mode.find("multiAngle") != -1:
            num += 1
        if self.input_mode.find("edge") != -1:
            num += 1
        if self.input_mode.find("sub") != -1:
            num += 1
        return num

    def methodsName_combine(self, ):
        if self.model_type == "spaceAndTime":
            if self.compare_mode == "model_cp":
                returnstr = self.model_method + "(" + self.input_mode + "_" + self.lossFunction_method + "_pre" + str(
                    self.preImg_num) + ")"
            elif self.compare_mode == "inputMode_cp":
                returnstr = self.input_mode + "(" + self.model_method + "_" + self.lossFunction_method + "_pre" + str(
                    self.preImg_num) + ")"
            elif self.compare_mode == "loss_cp":
                returnstr = self.lossFunction_method + "(" + self.model_method + "_" + self.input_mode + "_pre" + str(
                    self.preImg_num) + ")"
            print("modelMethod:", self.model_method, "\tinputMode:", self.input_mode, "\tlossfunction:",
                  self.lossFunction_method, "\tpreImg_num:", self.preImg_num)
        else:
            if self.compare_mode == "model_cp":
                returnstr = self.model_method + "(" + self.input_mode + "_" + self.lossFunction_method + ")"
            elif self.compare_mode == "inputMode_cp":
                returnstr = self.input_mode + "(" + self.model_method + "_" + self.lossFunction_method + ")"
            elif self.compare_mode == "loss_cp":
                returnstr = self.lossFunction_method + "(" + self.model_method + "_" + self.input_mode + ")"
            print("modelMethod:", self.model_method, "\tinputMode:", self.input_mode, "\tlossfunction:",
                  self.lossFunction_method)
        return returnstr
