import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F


class Load_Dataset(Dataset):
    def __init__(self, dataset, configs):
        super(Load_Dataset, self).__init__()
        udt_list = []  # 用于存储所有 udt_tensor 的列表
        result_list = []  # 用于存储所有用户的得分

        for gid in dataset.keys():
            gdata = dataset.get(gid)
            uids = list(gdata.keys())  # 获取当前 gid 下的所有 uid
            # 遍历每个 uid
            for uid in uids:
                udata = gdata.get(uid)
                udata_tensor = torch.tensor(udata['data'], dtype=torch.float32)
                utime_tensor = torch.tensor(udata['time'], dtype=torch.float32)
                speak_tensor = torch.tensor(udata['len_speak'], dtype=torch.float32)
                move_tensor = torch.tensor(udata['len_move'], dtype=torch.float32)
                result = udata['result']

                # 获取原始数据张量的形状
                original_shape = udata_tensor.shape
                # 计算填充后的长度
                pad_len = configs.max_len - original_shape[-1]
                # 使用 pad 函数进行填充，将数据张量的最后一个维度进行填充
                udata_tensor_padded = F.pad(udata_tensor, (0, pad_len), value=0)
                utime_tensor_padded = F.pad(utime_tensor, (0, pad_len), value=0)
                speak_tensor_padded = F.pad(speak_tensor, (0, pad_len), value=0)
                move_tensor_padded = F.pad(move_tensor, (0, pad_len), value=0)

                # 初始化一个包含 6 个特征的张量，数据维度为 max_len
                combined_tensor = torch.zeros((12, configs.max_len), dtype=torch.float32)

                # 将当前 uid 的 data 和 time 放入前两个特征
                combined_tensor[0] = udata_tensor_padded
                combined_tensor[1] = utime_tensor_padded
                combined_tensor[2] = speak_tensor_padded
                combined_tensor[3] = move_tensor_padded

                other_index = 0
                # 将同 gid 的其他 uid 的 data 和 time 加入到特征中
                for i, other_uid in enumerate(uids):
                    if other_uid != uid:
                        other_udata = gdata[other_uid]
                        other_udata_tensor = torch.tensor(other_udata['data'], dtype=torch.float32)
                        other_utime_tensor = torch.tensor(other_udata['time'], dtype=torch.float32)
                        other_speak_tensor = torch.tensor(other_udata['len_speak'], dtype=torch.float32)
                        other_move_tensor = torch.tensor(other_udata['len_move'], dtype=torch.float32)

                        other_original_shape = other_udata_tensor.shape
                        pad_len_other = configs.max_len - other_original_shape[-1]

                        # 对其他 uid 的 data 和 time 进行填充
                        other_udata_tensor_padded = F.pad(other_udata_tensor, (0, pad_len_other), value=0)
                        other_utime_tensor_padded = F.pad(other_utime_tensor, (0, pad_len_other), value=0)
                        other_speak_tensor_padded = F.pad(other_speak_tensor, (0, pad_len_other), value=0)
                        other_move_tensor_padded = F.pad(other_move_tensor, (0, pad_len_other), value=0)

                        # 将填充后的 data 和 time 放入 combined_tensor 中
                        combined_tensor[4 + 4 * other_index] = other_udata_tensor_padded
                        combined_tensor[4 + 4 * other_index + 1] = other_utime_tensor_padded
                        combined_tensor[4 + 4 * other_index + 2] = other_speak_tensor_padded
                        combined_tensor[4 + 4 * other_index + 3] = other_move_tensor_padded

                        other_index += 1

                # 将 combined_tensor 添加到列表中
                udt_list.append(combined_tensor)
                result_list.append(result)

        # 使用 torch.stack() 函数沿着新添加的维度进行堆叠
        X_data = torch.stack(udt_list, dim=0)
        X_data = X_data.permute(0, 2, 1)  # 这样使排列方式为 (batch_size, seq_len, features)
        Y_data = torch.tensor(result_list, dtype=torch.float32)
        self.x_data = X_data
        self.y_data = Y_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs):
    train_dataset = torch.load(os.path.join(data_path, "train_data.pt"))
    val_dataset = torch.load(os.path.join(data_path, "val_data.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_data.pt"))

    train_dataset = Load_Dataset(train_dataset, configs)
    val_dataset = Load_Dataset(val_dataset, configs)
    test_dataset = Load_Dataset(test_dataset, configs)

    """这里后期修改：是否随机打乱、是否丢弃最后一个不完整batch、是否需要开子进程"""
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=configs.shuffle, drop_last=configs.drop_last,
                                                    num_workers=configs.num_workers)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=configs.shuffle, drop_last=configs.drop_last,
                                                  num_workers=configs.num_workers)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                                   shuffle=configs.shuffle, drop_last=configs.drop_last,
                                                   num_workers=configs.num_workers)
    return train_data_loader, val_data_loader, test_data_loader
