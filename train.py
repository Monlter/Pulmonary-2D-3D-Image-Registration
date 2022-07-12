import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.data_loader import Dataset_CBCT, Dataset_PCA
from torch.utils.data import DataLoader
import math
from tools.loss_tool import PCA_loss, Log_cosh, PCA_smoothL1Loss
from tools.config import get_args
import yaml
from functools import partial
from tools.instanceExam import InstanceExam
from tools.tool_functions import *
from torch.utils.tensorboard import SummaryWriter
# 加载各类模型
from Model import *


def val(Dataset_loader, net, loss_function, device):
    val_loss = 0
    with torch.no_grad():
        for i, (imgs, target) in enumerate(Dataset_loader):
            imgs = imgs.to(device)
            target = target.to(device)
            prediction = net(imgs)
            loss = loss_function(prediction, target)
            val_loss += loss
    return (val_loss / (i + 1))


def load_cfg(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def train(args, cfg):
    # 初始化
    exam_process_dict = cfg['EXAM_PROCESS']
    total_exam_num = len(exam_process_dict)
    cur_exam_num = 0

    # 模型方式
    model_methods = {
        "CNN": CNN_model.CNN_net,
        "Unet": partial(Unet_model.UNet_net, n_classes=3),
        "Resnet": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2]),
        "Resnet_outCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_outAttention="CBAM"),
        "Resnet_outSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_outAttention="SPA"),
        "Resnet_inCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM"),
        "Resnet_outSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_outAttention="SE"),
        "Resnet_inSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA"),
        "Resnet_inSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE"),
        "Resnet_allSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                 is_outAttention="SPA"),
        "Resnet_allCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                  is_outAttention="CBAM"),
        "Resnet_allSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                is_outAttention="SE"),
        "Resnet_inSPA_outCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                        is_outAttention="CBAM"),
        "Resnet_inCBAM_outSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                        is_outAttention="SPA"),
        "Resnet_inCBAM_outSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                       is_outAttention="SE"),
        "Resnet_inSPA_outSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                      is_outAttention="SE"),
        "Resnet_inSE_outSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                      is_outAttention="SPA"),
        "Resnet_inSE_outCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                       is_outAttention="CBAM"),
    }
    # 损失函数方式
    lossfunction_methods = {
        "MSE": PCA_loss,
        "Smooth_MSE": PCA_smoothL1Loss,
        "log_cosh": Log_cosh
    }
    # 数据加载方式
    dataset_process_methods = {
        "pca": Dataset_PCA,
        "cbct": Dataset_CBCT,
        "dvf": Dataset_CBCT
    }
    setup_seed(12)
    # 超参数设定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    loss_wcoeff = torch.FloatTensor([2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)]).to(device)
    img_folder = os.path.join(args.root_path, args.img_folder)
    target_folder = os.path.join(args.root_path, args.CBCT_folder) if cfg["PREDICTION_MODE"] == "cbct" \
        else os.path.join(args.root_path, args.PCA_folder)

    # 进行各个exam
    for exam_cfg in exam_process_dict:
        exam_instance = InstanceExam(args, cfg, exam_cfg)
        # 生成log文件
        logger = get_logger(filename=os.path.join(exam_instance.log_dir, exam_instance.work_fileName + "_train.log"),
                            verbosity=1,
                            name=exam_instance.work_fileName)
        # 生成run文件
        writer = SummaryWriter(os.path.join(exam_instance.tensorboard_dir, exam_instance.work_fileName))

        model = model_methods[exam_instance.model_method](exam_instance.inChannel_num).to(device)
        loss_fn = lossfunction_methods[exam_instance.lossFunction_method](loss_wcoeff)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', verbose=True, patience=3)
        # 数据加载
        dataset = dataset_process_methods[exam_instance.prediction_mode](img_folder, target_folder,
                                                                         exam_instance.input_mode,
                                                                         exam_instance.model_type)
        test_size = int(len(dataset) * args.val_ratio)
        train_size = int(len(dataset) - test_size)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset,
                                                                    lengths=[train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(12))
        train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

        # 训练参数设定
        logger.info(
            "DATASET:" + str(exam_instance.dataset)
            + "\tMODEL_TYPE:" + str(exam_instance.model_type)
            + "\tPREDICTION_MODE:" + str(exam_instance.prediction_mode)
            + '\tCOMPARE_MODE:' + str(exam_instance.compare_mode)
            + '\tINPUT_MODE:' + str(exam_instance.input_mode)
            + "\tPREIMG_NUM:" + str(exam_instance.preImg_num)
            + "\tMODEL:" + str(exam_instance.model_method)
            + "\tLOSSFUNCTION:" + str(exam_instance.lossFunction_method)
        )
        logger.info("Epoch:" + str(args.EPOCH) + "\ttrain_dataset_num:" + str(train_size) + "\ttest_dataset_num:" + str(
            test_size))
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

            val_loss = val(test_data_loader, model, loss_fn, device)
            print('epoch:%d  train_loss:%.3f  test_loss:%.3f' % (epoch, loss_mse.item(), val_loss.item()))
            scheduler.step(loss_mse)
            logger.info(
                'Epoch:[{}/{}]\t train_loss={:.3f}\t test_loss={:.3f}'.format((epoch + 1), args.EPOCH, loss_mse.item(),
                                                                              val_loss.item()))
            writer.add_scalars("train_progress", {"train_loss": loss_mse.item(), "val_loss": val_loss.item()})

            if (epoch + 1) % args.EPOCH == 0:
                cur_ckpt_dir = os.path.join(exam_instance.ckpt_dir,exam_instance.work_fileName)
                os.makedirs(cur_ckpt_dir, exist_ok=True)
                cur_ckpt_file_name = os.path.join(cur_ckpt_dir, str(epoch + 1) + ".pth")
                torch.save(model.state_dict(), cur_ckpt_file_name)

        logger.info('finish training!')
        logging.shutdown()
        writer.close()


if __name__ == '__main__':
    args = get_args()
    cfg = load_cfg(yaml_path="./tools/cfg/pca_space.yaml")
    train(args, cfg)
