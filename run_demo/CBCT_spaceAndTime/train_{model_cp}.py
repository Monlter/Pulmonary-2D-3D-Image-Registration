import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.data_loader import Dataset_CBCT
from torch.utils.data import DataLoader
import math
import os
import logging
from tools.loss_tool import PCA_loss, Log_cosh, PCA_smoothL1Loss
from tools.config import get_args
import numpy as np
import cv2
from tools.tool_functions import *
from tools import loss_tool, estimate_methods
from torch.utils.tensorboard import SummaryWriter
from tools.other import save_ct_split
# 加载各类模型
from Model import *

args = get_args()


def init_args(modelMethodName, inputModeName, lossFunctionMethodName):
    args.modelMethod = modelMethodName
    args.inputMode = inputModeName
    args.lossFunctionMethod = lossFunctionMethodName

    current_file = get_filename(__file__)
    cpName = get_cpName(current_file)
    args.cpName = cpName

    testName = get_testName(__file__)
    args.testName = testName
    print(testName)

    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    args.root_path = root_path

    workFileName = methodsName_combine(args)
    args.workFileName = workFileName


def val(Dataset_loader, net, loss_function, device):
    val_loss = 0
    MAD = 0
    NCC = 0
    SSIM = 0
    with torch.no_grad():
        for i, (imgs, target) in enumerate(Dataset_loader):
            imgs = imgs.to(device)
            target = target.to(device)
            prediction = net(imgs)
            loss = loss_function(prediction, target)
            val_loss += loss
            MAD += estimate_methods.MAE(target[0].cpu().detach().numpy(), prediction[0].cpu().detach().numpy())
            NCC += estimate_methods.NCC(target[0].cpu().detach().numpy(), prediction[0].cpu().detach().numpy())
            SSIM += estimate_methods.SSIM(target[0].cpu().detach().numpy(), prediction[0].cpu().detach().numpy())
    return (val_loss / (i + 1)), (MAD / (i + 1)), (NCC / (i + 1)), (SSIM / (i + 1))


def save_split(data, save_path, epoch):
    if len(data.shape) == 3:
        # CT_dcm shape= C,H,W
        ct_crossview = data[73, :, :]
        ct_coronalview = data[:, 100, :]
        ct_sagittalview = data[:, :, 83]
        save_ct_split.save_png(ct_crossview, save_path, str(epoch) + "_ct_crossview")
        save_ct_split.save_png(ct_coronalview, save_path, str(epoch) + "_ct_coronalview")
        save_ct_split.save_png(ct_sagittalview, save_path, str(epoch) + "_ct_sagittalview")
    elif len(data.shape) == 4:
        # DVF shape = 3,C,H,W
        pass
    else:
        raise ("暂时不支持该数据类型的（{}）切片".format(data.shape))


def train(modelMethodName=args.modelMethod, inputModeName=args.inputMode,
          lossFunctionMethodName=args.lossFunctionMethod):
    # 初始化
    in_channels = get_channelNum(inputModeName)
    # 模型方式
    model_methods = {
        "ConvLSTM": convLSTM_2D.ConvLSTM_CBCT(in_channels),
    }
    # 损失函数方式
    lossfunction_methods = {
        "MSE": loss_tool.CBCT_MSE,
    }

    init_args(modelMethodName, inputModeName, lossFunctionMethodName)
    setup_seed(12)

    # if not modelMethodName:
    #     modelMethodName = args.common_model_name
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")

    # 结果路径（"PCA/Out_result/Test_space/"）：——> 用于生成log和run文件夹
    out_result_dir = get_out_result_dir(args)
    # log文件夹
    log_dir = make_dir(os.path.join(out_result_dir, 'log/'))
    # run文件夹
    tensorboard_dir = make_dir(os.path.join(out_result_dir, 'run/'))
    # ct_split 文件夹
    ct_split_dir = make_dir(os.path.join(out_result_dir, 'ct_split/'))
    # 保存路径：——> 用于保存训练的权重文件
    save_dir = get_checkpoint_dir(args)
    # 生成log文件
    logger = get_logger(log_dir + args.workFileName + "_train.log", 1, args.workFileName)
    # 生成run文件
    writer = SummaryWriter(tensorboard_dir + args.workFileName)

    # 超参数设定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    batch_size = args.batch_size
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', verbose=True, patience=3)
    loss_fn = lossfunction_methods[lossFunctionMethodName]
    # 数据加载
    img_folder = os.path.join(args.root_path, args.img_folder)
    target_folder = os.path.join(args.root_path, args.ct_folder)  # label为DVF
    dataset = Dataset_CBCT(img_folder, target_folder, inputModeName, args.testName)
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
        "TestName:" + str(args.testName) + "\tdata_method:" + str(args.gen_pca_method) + "\tmodel:" + str(
            modelMethodName) + '\tdataMethod:' + str(inputModeName) + '\tloss_function:' + str(lossFunctionMethodName))
    logger.info("Epoch:" + str(args.EPOCH) + "\tdata_num:" + str(num_dataset) + "\tdata_ratio:" + str(args.val_ratio))
    logger.info("---" * 100)
    loss_epoch = []

    logger.info('start training!')
    for epoch in range(args.EPOCH):
        loss_mse = 0
        for i, (imgs, target) in enumerate(train_data_loader):
            imgs = imgs.to(device)
            target = target.to(device)
            prediction = model(imgs)
            loss_item = loss_fn(target, prediction)
            loss_mse += loss_item
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss_item.backward()  # 误差反向传播，计算参数更新值
            opt.step()
            print("(epoch:%d--step:%d)------->loss:%.3f" % (epoch, i, loss_item.item()))
        loss_mse = loss_mse / (i + 1)
        loss_epoch.append(loss_mse.cpu().detach().numpy())

        val_loss, MAD, NCC, SSIM = val(test_data_loader, model, loss_fn, device)
        print('epoch:%d  train_loss:%.3f  test_loss:%.3f' % (epoch, loss_mse.item(), val_loss.item()))
        scheduler.step(loss_mse)
        logger.info(
            'Epoch:[{}/{}]\t train_loss={:.3f}\t test_loss={:.3f}\t MAD={:.5f}\t NCC={:.5f}\t SSIM={:.5f}'.format(
                (epoch + 1), args.EPOCH, loss_mse.item(), val_loss.item(), MAD.item(), NCC.item(), SSIM.item()))
        writer.add_scalars("train_progress",
                           {"train_loss": loss_mse.item(), "val_loss": val_loss.item(), "MAD": MAD.item(), "NCC":
                           NCC.item(), "SSIM": SSIM.item()})
        save_split(prediction[0, ...].cpu().detach().numpy(), ct_split_dir, epoch + 1)

        if (epoch + 1) % args.EPOCH == 0:
            save_file_name = os.path.join(save_dir, str(epoch + 1) + ".pth")
            torch.save(model.state_dict(), save_file_name)

    logger.info('finish training!')
    logging.shutdown()
    writer.close()


if __name__ == '__main__':
    train(modelMethodName="ConvLSTM")
