import torch
from torch import nn
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        # 将one-hot encoding 部分送入编码器作为一个批量的词嵌入向量
        # output_size为词汇量，input_size是词向量的长度
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        # 定义最后线性变换的维度，即最后一个卷积层的通道数到所有词汇的映射
        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            # 是否共享编码器与解码器的权重，默认值为共享
            # 共享时需要保持隐藏单元数等于词嵌入的长度
            # 此时将预测的向量认为是词嵌入向量
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        # 对输入词嵌入进行dropout
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        # 输入到网络进行推断
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        # 将推断结果解码为词
        y = self.decoder(y)
        return y.contiguous()

