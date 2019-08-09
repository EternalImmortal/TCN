import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 定义实现因果卷积的类
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    # tensor.contiguous()会返回有连续内存的相同张量

    # 有些tensor并不是占用一整块内存，而是由不同的数据块组成

    # 而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，就能把tensor变成在内存中连续分布的形式。

    # 通过增加Padding方式对卷积后的张量做切片而实现因果卷积

# 残差模块，其中有两个一维卷积与恒等映射
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 定义第一个扩散卷积层，dilation=dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 根据第一个卷积层的输出与padding的大小实现因果卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # padding保证输出和输入的序列长度相等。但是卷积前和卷积之后的信道数可能不一样
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将卷积模块的所有组建通过Sequential方法依次堆叠在一起
        # 具体来说的话网络结构是一层一层的叠加起来的，nn库里有一个类型：叫做Sequential序列，它是一个容器类，可以在里面添加一些基本的模块。
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 正如先前提到的卷积前与卷积后的通道数不一定相同
        # 所以如果通道数不一样，那么需要对输入x做一个逐元素的一维卷积
        # 以使得它的维度与前面两个卷积相等。
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # 初始化方法是从均值为0，标准差为0.01的正态分布采样。
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        self.net = self.net.double()
        out = self.net(x.double())
        # 残差模块
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 定义时间卷积模块
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        # num_channels为各层卷积运算的输出通道≠数或卷积核数量
        # num_channels的长度即需要执行的卷积层数量
        # 扩张系数若能随着网络层级的增加而成指数增加，
        # 则可以增大感受野并不丢弃任何输入序列的元素
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 从num_channels中抽取每一个残差模块的输入通道数与输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
