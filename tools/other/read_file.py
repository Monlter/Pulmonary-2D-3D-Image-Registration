import os
from tools.tool_functions import *

if __name__ == '__main__':
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    model_folder = os.path.join(root_path,"Dataset\Test_9dvf\Output\dvf")
    dvf_folder = os.path.join(root_path,"Dataset\Test_9dvf\Output\dvf\CNN(origin_MSE)")
    input_file = os.path.join(root_path,"Dataset\Test_9dvf\Output\input_list.txt")
    save_file = os.path.join(root_path,"Dataset\Test_9dvf\Output\out_list.txt")
    model_list = os.listdir(model_folder)
    dvf_list = os.listdir(dvf_folder)

    # 输入文件
    with open(input_file, 'w') as f1:
        for model_name in model_list:
            for dvf in dvf_list:
                f1.write(os.path.join(root_path,"Dataset\Test_9dvf\Output\dvf",model_name, dvf))
                f1.write('\n')

    # 输出文件
    with open(save_file, 'w') as f2:
        for model_name in model_list:
            out_dir = os.path.join(root_path,"Dataset\Test_9dvf\Output\CT",model_name)
            if not os.path.exists(out_dir):
               os.makedirs(out_dir)
            for dvf in dvf_list:
                save_name = "predict_ct_"+dvf[-1]
                f2.write(os.path.join(root_path,"Dataset\Test_9dvf\Output\CT", model_name, save_name))
                f2.write('\n')
