import os

if __name__ == '__main__':
    model_folder = r"G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\dvf"
    dvf_folder = r"G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\dvf\CNN(origin_MSE)"
    input_file = r"G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\input_list.txt"
    save_file = r"G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\out_list.txt"
    model_list = os.listdir(model_folder)
    dvf_list = os.listdir(dvf_folder)

    # 输入文件
    with open(input_file, 'w') as f1:
        for model_name in model_list:
            for dvf in dvf_list:
                f1.write(os.path.join("G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\dvf",model_name, dvf))
                f1.write('\n')

    # 输出文件
    with open(save_file, 'w') as f2:
        for model_name in model_list:
            out_dir = os.path.join("G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\CT",model_name)
            if not os.path.exists(out_dir):
               os.makedirs(out_dir)
            for dvf in dvf_list:
                save_name = "predict_ct_"+dvf[-1]
                f2.write(os.path.join("G:\Monlter\PCA\Pulmonary-2D-3D-Image-Registration\Dataset\Test_9dvf\Output\CT", model_name, save_name))
                f2.write('\n')
