import os.path

import torch
import numpy as np
from einops import rearrange


class Normalization:
    def __init__(self, data_path):
        self.lms_mean = torch.from_numpy(np.load(os.path.join(data_path, 'lms-mean.npy')).astype(np.float32))
        self.lms_std = torch.from_numpy(np.load(os.path.join(data_path, 'lms-std.npy')).astype(np.float32))
        self.pan_mean = torch.from_numpy(np.load(os.path.join(data_path, 'pan-mean.npy')).astype(np.float32))
        self.pan_std = torch.from_numpy(np.load(os.path.join(data_path, 'pan-std.npy')).astype(np.float32))

        # torch.cat
        self.input_mean = torch.cat([self.lms_mean, self.pan_mean], dim=0)
        self.input_std = torch.cat([self.lms_std, self.pan_std], dim=0)

        # self.input_mean = np.concatenate((self.lms_mean, self.pan_mean), axis=0)
        # self.input_std = np.concatenate((self.lms_std, self.pan_std), axis=0)

        self.gt_mean = torch.from_numpy(np.load(os.path.join(data_path, 'gt-mean.npy')).astype(np.float32))
        self.gt_std = torch.from_numpy(np.load(os.path.join(data_path, 'gt-std.npy')).astype(np.float32))

    def input_norm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = (data - self.input_mean) / self.input_std
        return rearrange(data, 'b h w c->b c h w')

    def gt_norm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = (data - self.gt_mean) / self.gt_std
        return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = data * self.gt_std + self.gt_mean
        return rearrange(data, 'b h w c->b c h w')


if __name__ == '__main__':
    import torch

    # 创建两个2x3的张量
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])

    # 使用torch.cat将这两个张量沿着第0维（行）连接起来
    result = torch.cat([tensor1, tensor2], dim=0)

    print(result)
