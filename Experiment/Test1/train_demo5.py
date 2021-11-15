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
# ¼ÓÔØ¸÷ÀàÄ£ÐÍ
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


def train(modelMethodName=args.common_model_name, dataMethodName=args.common_data_name,
          lossFuntionMethodName=args.common_lossfunction_name, imgFolder=args.img_folder):
    # ³õÊ¼»¯
    in_channels = get_dataMethod_num(dataMethodName)
    # Ä£ÐÍ·½Ê½
    model_methods = {
        "CNN": CNN_model.CNN_net(in_channels),
        "Unet": Unet_model.UNet_net(in_channels, 3),
        "Resnet": Resnet_attention.resnet(in_channels),
        "Resnet_Triplet": Resnet_Triplet_atttention.resnet(in_channels, is_Triplet=True),
        "Resnet_CBAM": Resnet_attention.resnet(in_channels, is_CBAM=True),
        "Resnet_dilation": Resnet_attention.resnet(in_channels, dilation=3),
        "Resnet_CBAM_dilation": Resnet_attention.resnet(in_channels, is_CBAM=True, dilation=3)
    }
    # ËðÊ§º¯Êý·½Ê½
    lossfunction_methods = {
        "MSE": PCA_loss,
        "Smooth_MSE": PCA_smoothL1Loss,
        "log_cosh": Log_cosh
    }
    # »ñÈ¡µ±Ç°ÑµÁ·µÄÊµÑéºÅ
    file_name = get_filename(__file__)
    num_cp = get_fileNum(file_name)
    testName = get_testName(__file__)

    setup_seed(12)

    if not modelMethodName:
        modelMethodName = args.common_model_name
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFuntionMethodName)
    if imgFolder == args.img_folder:
        workFileName = "raw_"+workFileName
    elif imgFolder.split("_")[-1] == "noise":
        workFileName = "noise_" + workFileName
    elif imgFolder.split("_")[-1] == "all":
        workFileName = "all_" + workFileName
    # ÊµÑéÂ·¾¶£¨"PCA/Experiment/Test1/PCA_origin/"£©£º¡ª¡ª> ÓÃÓÚÉú³ÉlogºÍrunÎÄ¼þ¼Ð
    experiment_dir = get_experimentDir(num_cp, root_path, testName, args.gen_pca_method)
    # logÎÄ¼þ¼Ð
    log_dir = make_dir(os.path.join(experiment_dir, 'log/'))
    # runÎÄ¼þ¼Ð
    tensorboard_dir = make_dir(os.path.join(experiment_dir, 'run/'))
    # ±£´æÂ·¾¶£º¡ª¡ª> ÓÃÓÚ±£´æÑµÁ·µÄÈ¨ÖØÎÄ¼þ
    save_dir = get_savedir(num_cp, root_path, testName, args.gen_pca_method, workFileName)
    # Éú³ÉlogÎÄ¼þ
    logger = get_logger(log_dir + workFileName + "_train.log", 1, workFileName)
    # Éú³ÉrunÎÄ¼þ
    writer = SummaryWriter(tensorboard_dir + workFileName)

    # ³¬²ÎÊýÉè¶¨
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    batch_size = args.batch_size
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', verbose=1, patience=3)
    wcoeff = torch.FloatTensor([2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)]).to(device)
    loss_fn = lossfunction_methods[lossFuntionMethodName](wcoeff)
    # Êý¾Ý¼ÓÔØ
    img_folder = os.path.join(root_path, imgFolder)
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

    # ÑµÁ·²ÎÊýÉè¶¨
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
            opt.zero_grad()  # Çå¿ÕÉÏÒ»²½²ÐÓà¸üÐÂ²ÎÊýÖµ
            loss_item.backward()  # Îó²î·´Ïò´«²¥£¬¼ÆËã²ÎÊý¸üÐÂÖµ
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
    recode_progressNum(1)
    train(modelMethodName="Resnet", imgFolder=os.path.join(os.path.dirname(args.img_folder), "projections_noise"))
    train(modelMethodName="Resnet", imgFolder=args.img_folder)
    train(modelMethodName="Resnet", imgFolder=os.path.join(os.path.dirname(args.img_folder), "projections_all"))
    recode_progressNum(2)
    train(modelMethodName="CNN", imgFolder=os.path.join(os.path.dirname(args.img_folder), "projections_noise"))
    train(modelMethodName="CNN", imgFolder=args.img_folder)
    train(modelMethodName="CNN", imgFolder=os.path.join(os.path.dirname(args.img_folder), "projections_all"))
    recode_progressNum(3)
    train(modelMethodName="Unet", imgFolder=os.path.join(os.path.dirname(args.img_folder), "projections_noise"))
    train(modelMethodName="Unet", imgFolder=args.img_folder)
    train(modelMethodName="Unet", imgFolder=os.path.join(os.path.dirname(args.img_folder), "projections_all"))
