class Config(object):
    max_len = 1000  # 个人行为序列最大长度
    num_classes = 22  # 行为分类数量
    len_dfeature = 2  # 离散手工特征维度

    n_layers = 1  # encoder层数
    batch_size = 16  # 一个batch处理数据量

    c_in = 1  # 输入数据的特征维度
    d_model = 512  # 嵌入层输出的维度

    dropout = 0.1
    shuffle = False  # data_loader是否打乱顺序
    drop_last = False  # data_loader是否丢弃最后一个不完整batch
    num_workers = 0  # data_loader子进程数量

    d_k = 64  # 自注意力k的维度
    d_v = 64  # 自注意力v的维度
    n_heads = 8  # 自注意力头的数量
    d_ff = 2048  # 前馈层中间维度

    lr = 0.0001  # 学习率
    epoch = 100  # 轮次
    temperature = 0.3  # 温度
    patience = 100  # 早停轮次

    beta1 = 0.9
    beta2 = 0.99

    alpha_ability = 0.3  # 能力权重，越大越重视问题解决能力，而非合作能力

    def __init__(self):
        pass
