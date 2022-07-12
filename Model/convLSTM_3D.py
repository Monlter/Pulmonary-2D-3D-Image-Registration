import torch.nn as nn
import torch


# convLSTM_2D:
class ConvLSTMCell_3D(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell_3D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # 上一状态的H和C

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)  # 将输出划分成4个通道为hidden_dim大小的输出
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        depth, height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device),
            # B,S,C,H,W
            torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device))


class ConvLSTM_3D(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM_3D, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        # self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell_3D(input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             kernel_size=self.kernel_size[i],
                                             bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        # if not self.batch_first:
        #     # (t, b, c, h, w) -> (b, t, c, h, w)
        #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)

        b, _, _, d, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()  # 如果没有hidden_state将会报错
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(d, h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # t
        cur_layer_input = input_tensor

        # 多层结构
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]  # 获取h和c
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])  # 当前层的h_0和c_0输入为上一层的h和c输出
                # h和c.shape:[B,hidden_size,H,W]
                output_inner.append(h)  # output_inner.shape:[seq_len,B,hidden_size,H,W]

            layer_output = torch.stack(output_inner, dim=1)  # layer_output.shape:[B,seq_len,hidden_size,H,W]
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)  # layer_output_list.shape:[num_layers,B,seq_len,hidden_size,H,W]
            last_state_list.append([h, c])  # last_state_list.shape:[num_layers,2,B,hidden_size,H,W]

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTM_DVF(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=3, kernel_size=(3, 3, 3), num_layers=2,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM_DVF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.depth_matcher = nn.Conv2d(input_dim, 150, kernel_size=3, padding=1, stride=1)
        self.convlstm_3D = ConvLSTM_3D(1, hidden_dim, kernel_size, num_layers,
                                       batch_first, bias, return_all_layers)
        self.flow = nn.Conv3d(hidden_dim, output_dim, kernel_size=3, padding=1, stride=1)
        # self.batch_norm = nn.BatchNorm3d()

    def forward(self, input_tensor):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        seq_len = input_tensor.size(1)
        fake_CBCT = []
        for t in range(seq_len):
            fake_CBCT.append(self.depth_matcher(input_tensor[:, t, ...]))
        fake_CBCT_seq = torch.stack(fake_CBCT, dim=1).unsqueeze(dim=2)
        layer_output_list, last_state_list = self.convlstm_3D(fake_CBCT_seq)
        x = layer_output_list[-1][:, -1, ...]  # 取出num_layer的最后一层中的最后一个seq_len的结果
        flow = self.flow(x)
        # flow = self.batch_norm(flow)
        return flow
