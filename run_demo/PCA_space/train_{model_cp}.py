import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.data_loader import Dataset_PCA
from torch.utils.data import DataLoader
import math
import time
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


def init_args(modelMethodName, inputModeName,lossFunctionMethodName):
    args.modelMethod = modelMethodName
    args.inputMode = inputModeName
    args.lossFunctionMethod = lossFunctionMethodName

    current_file = get_filename(__file__)   # 获取path最后的文件名
    cpName = get_cpName(current_file)       # 获取{}内的内容
    args.cpName = cpName

    testName = get_testName(__file__)
    args.testName = testName

    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    args.root_path = root_path

    workFileName = methodsName_combine(args)
    args.workFileName = workFileName

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


def train(modelMethodName=args.modelMethod, inputModeName=args.inputMode,lossFunctionMethodName=args.lossFunctionMethod):
    # 初始化
    in_channels = get_channelNum(inputModeName)
    # 模型方式
    model_methods = {
        "CNN": CNN_model.CNN_net(in_channels),
        "Unet": Unet_model.UNet_net(in_channels, 3),
        "Resnet": Resnet_attention.resnet(in_channels,layers=[2,2,2,2]),
        "Resnet_outTriplet": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2],  is_outAttention="Triplet"),
        "Resnet_outCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="CBAM"),
        "Resnet_dilation": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], dilation=3),
        "Resnet_outSPA_dilation": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="SPA", dilation=3),
        "Resnet_outCBAM_dilation": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="CBAM", dilation=3),
        "Resnet_outSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="SPA"),
        "Resnet_inCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM"),
        "Resnet_outSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="SE"),
        "Resnet_inSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA"),
        "Resnet_inSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE"),
        "Resnet_allSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA", is_outAttention="SPA"),
        "Resnet_allCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM", is_outAttention="CBAM"),
        "Resnet_allSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE",is_outAttention="SE"),
        "Resnet_inSPA_outCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA", is_outAttention="CBAM"),
        "Resnet_inCBAM_outSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM", is_outAttention="SPA"),
        "Resnet_inCBAM_outSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",is_outAttention="SE"),
        "Resnet_inSPA_outSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA",is_outAttention="SE"),
        "Resnet_inSE_outSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE", is_outAttention="SPA"),
        "Resnet_inSE_outCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE", is_outAttention="CBAM"),
    }
    # 损失函数方式
    lossfunction_methods = {
        "MSE": PCA_loss,
        "Smooth_MSE": PCA_smoothL1Loss,
        "log_cosh": Log_cosh
    }
    init_args(modelMethodName, inputModeName, lossFunctionMethodName)

    setup_seed(12)

    if not modelMethodName:
        modelMethodName = args.common_model_name
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    out_result_dir = get_out_result_dir(args)
    # log文件夹
    log_dir = make_dir(os.path.join(out_result_dir, 'log/'))
    # run文件夹
    tensorboard_dir = make_dir(os.path.join(out_result_dir, 'run/'))
    # 保存路径：——> 用于保存训练的权重文件
    save_dir = get_checkpoint_dir(args)
    logger = get_logger(log_dir + args.workFileName + "_train.log", 1, args.workFileName)

    writer = SummaryWriter(tensorboard_dir + args.workFileName)


    # 超参数设定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    batch_size = args.batch_size
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', verbose=1, patience=3)
    wcoeff = torch.FloatTensor([2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)]).to(device)
    loss_fn = lossfunction_methods[lossFunctionMethodName](wcoeff)
    # 数据加载
    img_folder = os.path.join(root_path, args.img_folder)
    target_folder = os.path.join(root_path, args.target_folder)
    PCA_all_folder = os.path.join(root_path, args.PCA_all_folder)
    dataset = Dataset_PCA(img_folder, target_folder, PCA_all_folder, inputModeName, args.testName)
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

        if (epoch + 1) % args.EPOCH == 0:
            save_file_name = str(save_dir + str(epoch + 1) + ".pth")
            torch.save(model.state_dict(), save_file_name)

    logger.info('finish training!')
    logging.shutdown()
    writer.close()


if __name__ == '__main__':
    print("-"*100)
    print("progress_1", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="CNN")
    print("-" * 100)
    print("progress_2", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Unet")
    print("-" * 100)
    print("progress_3", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    train(modelMethodName="Resnet")
    print("-" * 100)
    # print("progress_4", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    # train(modelMethodName="Resnet_outTriplet")
    print("-" * 100)
    print("progress_5", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    train(modelMethodName="Resnet_outCBAM")
    print("-" * 100)
    print("progress_6", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    train(modelMethodName="Resnet_dilation")
    print("-" * 100)
    print("progress_7", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    train(modelMethodName="Resnet_outSPA_dilation")
    print("-" * 100)
    print("progress_7", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_outCBAM_dilation")
    print("-" * 100)
    print("progress_9", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_outSPA")
    print("-" * 100)
    print("progress_10", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inCBAM")
    print("-" * 100)
    print("progress_11", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inSPA")
    print("-" * 100)
    print("progress_12", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_allSPA")
    print("-" * 100)
    print("progress_13", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_allCBAM")
    print("-" * 100)
    print("progress_14", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inSPA_outCBAM")
    print("-" * 100)
    print("progress_15", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inCBAM_outSPA")
    print("-" * 100)
    print("progress_16", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_outSE")
    print("-" * 100)
    print("progress_17", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inSE")
    print("-" * 100)
    print("progress_18", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_allSE")
    print("-" * 100)
    print("progress_19", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inSPA_outSE")
    print("-" * 100)
    print("progress_20", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inSE_outSPA")
    print("-" * 100)
    print("progress_21", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inCBAM_outSE")
    print("-" * 100)
    print("progress_22", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train(modelMethodName="Resnet_inSE_outCBAM")










