import math
import os
import random
import time

import numpy as np
import torch
import torch.utils.data as data
from jacksung.utils.multi_task import MultiTasks
from jacksung.utils.time import Stopwatch
import jacksung.utils.fastnumpy as fnp
from tqdm import tqdm


class Benchmark(data.Dataset):
    def __init__(self, data_path, train=False, repeat=1, read_gt=True):
        super(Benchmark, self).__init__()
        self.train = train
        self.repeat = repeat
        self.data_path = data_path
        self.read_gt = read_gt
        self.data = {}
        # self.count = np.load(os.path.join(data_path, 'gt.npy')).shape[0]
        self.count = 5
        self.total_bar = tqdm(total=self.count, desc='Loading data')
        m = MultiTasks(10)

        for load_index in range(0, self.count):
            m.add_task(load_index, self.work, [load_index])
        m.execute_task(print_percent=False)

    def work(self, load_index):
        self.data[load_index] = self.load_data(load_index)
        self.total_bar.update()

    def __len__(self):
        if self.train:
            return self.count * self.repeat
        else:
            return self.count

    def load_data(self, loaded_index):
        lms_np = np.load(os.path.join(self.data_path, 'lms.npy')).astype(np.float32)
        lms_np = lms_np[loaded_index, :, :, :]
        pan_np = np.load(os.path.join(self.data_path, 'pan.npy')).astype(np.float32)
        pan_np = pan_np[loaded_index, :, :, :]
        input_np = np.concatenate((lms_np, pan_np), axis=0)

        if self.read_gt:
            gt_np = np.load(os.path.join(self.data_path, 'gt.npy')).astype(np.float32)
            gt_np = gt_np[loaded_index, :, :, :]
        else:
            gt_np = None

        r = [input_np, gt_np]
        r = [e for e in r if e is not None]
        return r

    def __getitem__(self, idx):
        idx = idx % self.count
        result = self.data[idx]
        return result


if __name__ == '__main__':
    # data_path = r'E:\pycode\Pansharpening\data\GF2'
    # dataset = Benchmark(data_path, train=True)
    npy1 = np.load(r'E:\pycode\Pansharpening\data\GF2\Valid\lms.npy')
    pass
