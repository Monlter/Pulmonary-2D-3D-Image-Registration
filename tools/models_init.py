from Model import *
from tools.loss_tool import PCA_loss, Log_cosh, PCA_smoothL1Loss
from functools import partial

def optional_init():
    # 模型方式
    # 模型方式
    model_methods = {
        "Test1": CNN_model.CNN_net,
        "Test2": partial(Unet_model.UNet_net, n_classes=3),
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
        "ConvLSTMLiner_h10_l2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=10, num_layers=2),
        "ConvLSTMLiner_h20_l2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=2),
        "ConvLSTMLiner_h30_l2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=30, num_layers=2),
        "ConvLSTMLiner_h40_l2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=40, num_layers=2),
        "ConvLSTMLiner_h50_l2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=50, num_layers=2),
        "ConvLSTMLiner_h20_l4": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=4),
        "ConvLSTMLiner_h20_l6": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=6),
        "ConvLSTMLiner_h20_l8": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=8),
        "ConvLSTMLiner_h20_l10": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=10),

        "ConvLSTMLiner_h30_l2_pA2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=30, num_layers=2,
                                            is_pooling=["Avgpool", "2"]),
        "ConvLSTMLiner_h30_l2_pC2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=30, num_layers=2,
                                            is_pooling=["Convpool", "2"]),
        "ConvLSTMLiner_h30_l2_pM2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=30, num_layers=2,
                                            is_pooling=["Maxpool", "2"]),

        "ConvLSTMLiner_h20_l2_pA2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=2,
                                            is_pooling=["Avgpool", "2"]),
        "ConvLSTMLiner_h20_l2_pC2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=2,
                                            is_pooling=["Convpool", "2"]),
        "ConvLSTMLiner_h20_l2_pM2": partial(convLSTM_2D.ConvLSTM_Liner, hidden_dim=20, num_layers=2,
                                            is_pooling=["Maxpool", "2"]),

    }
    # 损失函数方式
    lossfunction_methods = {
        "MSE": PCA_loss,
        "Smooth_MSE": PCA_smoothL1Loss,
        "log_cosh": Log_cosh
    }
    return model_methods, lossfunction_methods