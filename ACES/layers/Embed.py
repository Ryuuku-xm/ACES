import torch
import torch.nn as nn
import math
from config_files.parameter import Config


# 总嵌入
class DataEmbedding(nn.Module):
    def __init__(self, embed_type='coopeformer'):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding()
        self.temporal_embedding = TemporalEmbedding(embed_type=embed_type)
        self.dfeature_embedding = DfeatureEmbedding()

        self.dropout = nn.Dropout(p=Config.dropout)

    def forward(self, x):
        x_type = x[:, :, :1].squeeze(dim=2)
        x_timeandtype = x[:, :, :2]
        x_feature_d = x[:, :, 2:4]

        x_type = self.value_embedding(x_type)
        x_time = self.temporal_embedding(x_timeandtype)
        x_dfeature = self.dfeature_embedding(x_feature_d)

        # 合并所有嵌入
        x = x_type + x_time + x_dfeature
        return self.dropout(x)


# 行为类型序列嵌入
class TokenEmbedding(nn.Module):
    def __init__(self, num_classes=Config.num_classes, d_model=Config.d_model):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(num_classes, d_model)

    def forward(self, x):
        x = x.int()
        token_embeddings = self.token_embedding(x)
        return token_embeddings


# 时间序列嵌入
class TemporalEmbedding(nn.Module):
    def __init__(self, embed_type):
        super(TemporalEmbedding, self).__init__()
        self.embed_type = embed_type

        if embed_type == 'THP':
            Embed = FixedEmbedding
        elif embed_type == 'TimeF':
            Embed = TimeFeatureEmbedding
        else:
            Embed = TityEmbedding

        self.embed = Embed()

    def forward(self, x):
        x = x.long()

        if self.embed_type in ['THP', 'TimeF']:
            x = x[:, :, 1:].squeeze(dim=2)

        x = self.embed(x)
        return x


# 离散手工特征嵌入
class DfeatureEmbedding(nn.Module):
    def __init__(self, num_features=Config.len_dfeature, d_model=Config.d_model):
        super(DfeatureEmbedding, self).__init__()
        self.linear = nn.Linear(num_features, d_model)

    def forward(self, x):
        x = x.float()  # 确保输入为浮点型
        feature_embeddings = self.linear(x)  # 通过线性层嵌入
        return feature_embeddings


class TityEmbedding(nn.Module):
    def __init__(self, num_classes=Config.num_classes, d_model=Config.d_model):
        super(TityEmbedding, self).__init__()

        # 初始化正弦/余弦编码权重
        self.position_embedding = FixedEmbedding(c_in=num_classes, d_model=d_model)

        # 为每种行为类型定义可学习缩放系数
        self.ac = nn.Parameter(torch.randn(num_classes, 1))  # 对应于每种行为的可学习缩放系数

        # 线性层，用于将连续值转换为嵌入维度
        self.linear_transform = nn.Linear(1, d_model)

    def forward(self, x):
        actions = x[:, :, 0:1].long().squeeze(dim=2)  # 行为种类
        timestamps = x[:, :, 1:2].squeeze(dim=2)  # 时间序列

        # 计算事件绝对位置编码
        position_embeddings = self.position_embedding(actions)

        # 计算相对时间差
        relative_time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
        relative_time_diffs = torch.cat((torch.zeros_like(relative_time_diffs[:, :1]), relative_time_diffs), dim=1)

        # 使用缩放系数进行嵌入
        relative_time_embeddings = self.ac[actions] * relative_time_diffs.unsqueeze(dim=2)

        # 将相对时间嵌入转换为嵌入维度
        relative_time_embeddings = self.linear_transform(relative_time_embeddings)

        # 合并绝对位置编码和相对时间嵌入
        embeddings = position_embeddings + relative_time_embeddings

        return embeddings


class FixedEmbedding(nn.Module):
    def __init__(self, c_in=Config.c_in, d_model=Config.d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=Config.d_model, max_len=Config.max_len):
        super(PositionalEmbedding, self).__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 计算位置编码
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 添加批处理维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 返回对应输入序列长度的位置信息编码
        return self.pe[:, :x.size(1)]


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model=Config.d_model):
        super(TimeFeatureEmbedding, self).__init__()

        # 输入为分钟和秒的两个特征
        self.embed = nn.Linear(2, d_model, bias=False)

    def forward(self, x):
        # 将输入的秒数转换为分钟和秒
        minutes = x // 60  # 整数部分为分钟
        seconds = x % 60  # 余数部分为秒

        # 组合成一个新的张量，形状为 (batch_size, seq_len, 2)
        time_features = torch.stack((minutes, seconds), dim=-1)

        # 嵌入
        return self.embed(time_features)


# 废案——patch嵌入
class PatchEmbedding_c(nn.Module):
    def __init__(self, d_model, patch_size, max_len=Config.max_len):
        super(PatchEmbedding_c, self).__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # 计算窗口数量
        self.num_patches = (max_len + patch_size - 1) // patch_size

        # 线性层将每个 patch 映射到 d_model 维度
        self.projection = nn.Linear(patch_size * d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()

        # 计算每个 patch 的数量
        patch_size = self.patch_size
        num_patches = self.num_patches

        # 添加 padding 使序列长度变成 patch_size 的倍数
        if seq_len % patch_size != 0:
            padding_size = (patch_size - (seq_len % patch_size))
            x = torch.cat([x, torch.zeros(batch_size, padding_size, d_model, device=x.device)], dim=1)

        # 重新排列数据为 (batch_size, num_patches, patch_size * d_model)
        x = x.view(batch_size, num_patches, patch_size * d_model)

        # 对每个 patch 应用线性变换
        x = self.projection(x)

        return x
