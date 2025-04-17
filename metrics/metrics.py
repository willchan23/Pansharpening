import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
import math
# from pytorch_msssim import ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import R2Score
from torchmetrics.regression import MeanSquaredError
import importlib


def compute_mse(da_fc, da_true, lat_weight):
    error = da_fc - da_true
    # error = error ** 2
    error = rearrange(error ** 2, 'b c h w->b c w h') * lat_weight
    number = error.mean((-2, -1))
    return number.mean()


def compute_rmse(da_fc, da_true, lat_weight):
    error = da_fc - da_true
    # error = error ** 2
    error = rearrange(error ** 2, 'b c h w->b c w h') * lat_weight
    number = torch.sqrt(error.mean((-2, -1)))
    return number.mean()
    # number = torch.sqrt(error.mean())
    # return number


def compute_rr(da_fc, da_true, lat_weight):
    # true_mean = da_true.mean()
    true_mean = (rearrange(da_true, 'b c h w->b c w h') * lat_weight).mean()
    res = compute_mse(da_fc, da_true, lat_weight)
    tss = compute_mse(da_true, true_mean, lat_weight)
    rr = 1 - res / tss
    return rr


def compute_acc(da_fc, da_true, lat_weight, mean):
    fc_miuns_mean = rearrange(da_fc, 'b c h w->b h w c')
    fc_miuns_mean = fc_miuns_mean - mean
    fc_miuns_mean = rearrange(fc_miuns_mean, 'b h w c->b c w h')

    true_miuns_mean = rearrange(da_true, 'b c h w->b h w c')
    true_miuns_mean = true_miuns_mean - mean
    true_miuns_mean = rearrange(true_miuns_mean, 'b h w c->b c w h')
    acc_up = torch.sum((fc_miuns_mean * true_miuns_mean) * lat_weight)

    acc_down = torch.sqrt(torch.sum((fc_miuns_mean ** 2) * lat_weight) * torch.sum((true_miuns_mean ** 2) * lat_weight))
    acc = acc_up / acc_down
    return acc


########################################################################################
def compute_test_mse(da_fc, da_true, lat_weight):
    error = da_fc - da_true
    # error = error ** 2
    error = rearrange(error ** 2, 'h w->w h') * lat_weight
    number = error.mean((-2, -1))
    return number.mean()


def compute_test_rmse(da_fc, da_true, lat_weight):
    error = da_fc - da_true
    # error = error ** 2
    error = rearrange(error ** 2, 'h w->w h') * lat_weight
    number = torch.sqrt(error.mean((-2, -1)))
    return number.mean()
    # number = torch.sqrt(error.mean())
    # return number


def compute_test_rr(da_fc, da_true, lat_weight):
    # true_mean = da_true.mean()
    true_mean = (rearrange(da_true, 'h w->w h') * lat_weight).mean()
    res = compute_test_mse(da_fc, da_true, lat_weight)
    tss = compute_test_mse(da_true, true_mean, lat_weight)
    rr = 1 - res / tss
    return rr


def compute_test_acc(da_fc, da_true, lat_weight, mean):
    # fc_miuns_mean = rearrange(da_fc, 'h w->h w c')
    fc_miuns_mean = rearrange(da_fc - mean, 'h w -> w h')
    # fc_miuns_mean = rearrange(fc_miuns_mean, 'h w c->c w h')

    # true_miuns_mean = rearrange(da_true, 'c h w->h w c')
    true_miuns_mean = rearrange(da_fc - mean, 'h w -> w h')
    # true_miuns_mean = rearrange(true_miuns_mean, 'h w c->c w h')
    acc_up = torch.sum((fc_miuns_mean * true_miuns_mean) * lat_weight)

    acc_down = torch.sqrt(torch.sum((fc_miuns_mean ** 2) * lat_weight) * torch.sum((true_miuns_mean ** 2) * lat_weight))
    acc = acc_up / acc_down
    return acc


class Metrics:
    def __init__(self):
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

        self.weights_lat = np.arange(0, 65, 0.0625)
        self.weights_lat = np.cos(self.weights_lat * np.pi / 180)
        self.weights_lat = torch.from_numpy(1040 * self.weights_lat / np.sum(self.weights_lat))
        # self.mean = None
        self.mean = torch.from_numpy(
            np.load('/mnt/data1/czx/cldas_npy/new4v/2020/2020-total-mean.npy').astype(np.float32))
        # self.R2Score = R2Score()
        # /mnt/data1/czx/cldas_npy/new4v/2020/2020-total-mean.npy

    def calc_psnr(self, sr, hr):
        return self.psnr(sr, hr)

    def calc_ssim(self, sr, hr):
        return self.ssim(sr, hr)

    def calc_rmse(self, sr, hr):
        if sr.device != self.weights_lat.device:
            self.weights_lat = self.weights_lat.to(sr.device)
            self.mean = self.mean.to(sr.device)
        return compute_rmse(sr, hr, self.weights_lat)

    def calc_rr(self, sr, hr):
        if sr.device != self.weights_lat.device:
            self.weights_lat = self.weights_lat.to(sr.device)
            self.mean = self.mean.to(sr.device)
        return compute_rr(sr, hr, self.weights_lat)

    def calc_acc(self, sr, hr, v_index):
        mean = self.mean[v_index]
        return compute_acc(sr, hr, self.weights_lat, mean)

    ####################################################################################
    def calc_test_rmse(self, sr, hr):
        if sr.device != self.weights_lat.device:
            self.weights_lat = self.weights_lat.to(sr.device)
            self.mean = self.mean.to(sr.device)
        return compute_test_rmse(sr, hr, self.weights_lat)

    def calc_test_rr(self, sr, hr):
        if sr.device != self.weights_lat.device:
            self.weights_lat = self.weights_lat.to(sr.device)
            self.mean = self.mean.to(sr.device)
        return compute_test_rr(sr, hr, self.weights_lat)

    def calc_test_acc(self, sr, hr, v_index):
        mean = self.mean[v_index]
        return compute_test_acc(sr, hr, self.weights_lat, mean)


if __name__ == '__main__':
    m = Metrics()
    # preds = torch.rand(2, 3, 35, 55)
    # target = torch.rand(2, 3, 35, 55)
    # test_tensor = torch.rand(2, 3, 4, 1)
    # # m = Metrics()
    # # w = m.weights_lat[0k]
    # mean = torch.from_numpy(
    #     np.load(r'E:\pycode\cldas_geoan\CLDAS_GeoAN\temp\2020-total-mean.npy').astype(np.float32))[0]
    # a = test_tensor - mean
    # print()
    # print(m.calc_psnr(preds, target))
    # print(m.calc_ssim(preds, target))
    # print(m.calc_rmse(preds, target))
    # print(m.calc_rr(preds, preds))
    y_ = np.load(r"E:\temp\cldas_for\10月\y\2021020102-cldas-hourly.npy")
    target = np.load(r"E:\temp\cldas_for\10月\terget\2020020102-cldas-hourly.npy")

    y_PRS = y_[0, :, :]
    target_PRS = target[0, :, :]
    # psnr += float(m.calc_psnr(y_, target))
    # ssim += float(m.calc_ssim(y_, target))
    rmse_PRS = float(m.calc_test_rmse(y_PRS, target_PRS))
    acc_PRS = float(m.calc_test_acc(y_PRS, target_PRS, 0))
    rr_PRS = float(m.calc_test_rr(y_PRS, target_PRS))
    print()
