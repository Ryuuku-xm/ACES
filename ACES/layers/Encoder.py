import torch.nn as nn
from torch.nn import MultiheadAttention
from config_files.parameter import Config


# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=Config.d_model, d_ff=Config.d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


# Encoder单层
class EncoderLayer(nn.Module):
    def __init__(self, d_model=Config.d_model, n_heads=Config.n_heads):
        super(EncoderLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, n_heads)  # 使用已有的多头注意力
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈神经网络
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=Config.dropout)

    def forward(self, q_inputs, kv_inputs, kv_mask):
        attn_output, _ = self.multihead_attn(q_inputs, kv_inputs, kv_inputs, key_padding_mask=kv_mask)
        attn_output = self.dropout(attn_output)  # Dropout
        enc_outputs = self.layer_norm1(q_inputs + attn_output)  # 残差连接与层归一化
        enc_outputs = self.pos_ffn(enc_outputs)  # 前馈网络(conv+relu+conv+残差)
        return self.layer_norm2(enc_outputs)  # 再次进行层归一化

