import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.data_loader import Dataset_variable
from torch.utils.data import DataLoader
import math
import os
import logging
from tools.loss_tool import PCA_loss, Log_cosh, PCA_smoothL1Loss
from tools.config import get_args
import numpy as np

from tools.tool_functions import *
from torch.utils.tensorboard import SummaryWriter
# 加载各类模型
from Model import *

args = get_args()

def val(Dataset_loader, net, loss_function, device):
    val_loss = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(Dataset_loader):
            img = img.to(device)
            target = target.to(device)
            prediction = net(img)
            loss = loss_function(prediction, target)
            val_loss += loss
    return (val_loss / (i + 1))


def train(modelMethodName=args.common_model_name, dataMethodName=args.common_data_name, lossFuntionMethodName=args.common_lossfunction_name):
    # 初始化
    in_channels = get_dataMethod_num(dataMethodName)
    # 模型方式
    model_methods = {
        "CNN": CNN_model.CNN_net(in_channels),
        "Unet": Unet_model.UNet_net(in_channels, 3),
        "Resnet": Resnet_attention.resnet(in_channels),
        "Resnet_Triplet":Resnet_Triplet_atttention.resnet(in_channels,is_Triplet=True),
        "Resnet_CBAM":Resnet_attention.resnet(in_channels,is_CBAM=True),
        "Resnet_dilation":Resnet_attention.resnet(in_channels,dilation=3),
        "Resnet_CBAM_dilation": Resnet_attention.resnet(in_channels,is_CBAM=True, dilation=3)
    }
    # 损失函数方式
    lossfunction_methods = {
        "MSE": PCA_loss,
        "Smooth_MSE": PCA_smoothL1Loss,
        "log_cosh": Log_cosh
    }
    # 获取当前训练的实验号
    file_name = get_filename(__file__)
    num_cp = get_fileNum(file_name)
    testName = get_testName(__file__)

    setup_seed(12)

    if not modelMethodName:
        modelMethodName = args.common_model_name
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFuntionMethodName)
    # 实验路径（"PCA/Experiment/Test1/PCA_origin/"）：——> 用于生成log和run文件夹
    experiment_dir = get_experimentDir(num_cp, root_path, testName, args.gen_pca_method)
    # log文件夹
    log_dir = make_dir(os.path.join(experiment_dir, 'log/'))
    # run文件夹
    tensorboard_dir = make_dir(os.path.join(experiment_dir, 'run/'))
    # 保存路径：——> 用于保存训练的权重文件
    save_dir = get_savedir(num_cp, root_path, testName, args.gen_pca_method, workFileName)
    # 生成log文件
    logger = get_logger(log_dir + workFileName + "_train.log",1,workFileName)
    # 生成run文件
    writer = SummaryWriter(tensorboard_dir + workFileName)

    # 超参数设定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    batch_size = args.batch_size
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', verbose=1, patience=3)
    wcoeff = torch.FloatTensor([2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)]).to(device)
    loss_fn = lossfunction_methods[lossFuntionMethodName](wcoeff)
    # 数据加载
    img_folder = os.path.join(root_path, args.img_folder)
    target_folder = os.path.join(root_path, args.target_folder)
    PCA_all_folder = os.path.join(root_path, args.PCA_all_folder)
    dataset = Dataset_variable(img_folder, target_folder, PCA_all_folder, dataMethodName)
    num_dataset = len(dataset)
    test_size = int(len(dataset) * args.val_ratio)
    train_size = int(len(dataset) - test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset,
                                                                lengths=[train_size, test_size],
                                                                generator=torch.Generator().manual_seed(12))
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # 训练参数设定
    logger.info(
        "Experiment:" + str(testName) + "\tdata_method:" + str(args.gen_pca_method) + "\tmodel:" + str(
            modelMethodName) + '\tdataMethod:' + str(dataMethodName) + '\tloss_function:' + str(lossfunction_methods))
    logger.info("Epoch:" + str(args.EPOCH) + "\tdata_num:" + str(num_dataset) + "\tdata_ratio:" + str(args.val_ratio))
    logger.info("---" * 100)
    loss_epoch = []

    logger.info('start training!')
    for epoch in range(args.EPOCH):
        loss_mse = 0
        for i, (img, target) in enumerate(train_data_loader):
            img = img.to(device)
            target = target.to(device)
            prediction = model(img)
            loss_item = loss_fn(target, prediction)
            loss_mse += loss_item
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss_item.backward()  # 误差反向传播，计算参数更新值
            opt.step()
            print("(epoch:%d--step:%d)------->loss:%.3f" % (epoch, i, loss_item.item()))
        loss_mse = loss_mse / (i + 1)
        loss_epoch.append(loss_mse.cpu().detach().numpy())

        val_loss = val(test_data_loader, model, loss_fn, device)
        print('epoch:%d  train_loss:%.3f  test_loss:%.3f' % (epoch, loss_mse.item(), val_loss.item()))
        scheduler.step(loss_mse)
        logger.info(
            'Epoch:[{}/{}]\t train_loss={:.3f}\t test_loss={:.3f}'.format((epoch + 1), args.EPOCH, loss_mse.item(),
                                                                          val_loss.item()))
        writer.add_scalars("train_progress", {"train_loss": loss_mse.item(), "val_loss": val_loss.item()})

        if (epoch + 1) % 50 == 0:
            save_file_name = str(save_dir + str(epoch + 1) + ".pth")
            torch.save(model.state_dict(), save_file_name)

    logger.info('finish training!')
    logging.shutdown()
    writer.close()


if __name__ == '__main__':
    train("Resnet_Triplet")
