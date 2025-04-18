import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div

        channels_idx = list(range(inp_channels))
        random.shuffle(channels_idx)
        self.weight[channels_idx[0 * g:1 * g], 0, 1, 2] = 1.0  ## left
        self.weight[channels_idx[1 * g:2 * g], 0, 1, 0] = 1.0  ## right
        self.weight[channels_idx[2 * g:3 * g], 0, 2, 1] = 1.0  ## up
        self.weight[channels_idx[3 * g:4 * g], 0, 0, 1] = 1.0  ## down
        self.weight[channels_idx[4 * g:], 0, 1, 1] = 1.0  ## identity
        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='conv3'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        elif conv_type == 'common':
            self.shift_conv = nn.Conv2d(inp_channels, out_channels, kernel_size=1)
        elif conv_type == 'conv3':
            self.shift_conv = nn.Conv2d(inp_channels, out_channels, kernel_size=3, padding=1)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class ACT(nn.Module):
    def __init__(self):
        super(ACT, self).__init__()
        # self.act = nn.Mish()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x)


class DownBlock(nn.Module):
    def __init__(self, c_lgan, downscale=2):
        super(DownBlock, self).__init__()
        # self.down = nn.Conv2d(c_lgan, c_lgan * downscale, kernel_size=downscale, stride=downscale)
        # self.norm = Norm(c_lgan * downscale)
        # self.act = ACT()
        # self.conv = nn.Conv2d(c_lgan * downscale, c_lgan * downscale, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(c_lgan, c_lgan, kernel_size=downscale, stride=downscale)
        self.norm = Norm(c_lgan)
        self.act = ACT()
        self.conv = nn.Conv2d(c_lgan, c_lgan, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, c_lgan, downscale=2):
        super(UpBlock, self).__init__()
        # self.conv = nn.Conv2d(c_lgan * downscale, c_lgan * downscale, kernel_size=3, stride=1, padding=1)
        # self.norm = Norm(c_lgan * downscale)
        # self.act = ACT()
        # self.up = nn.ConvTranspose2d(c_lgan * downscale, c_lgan, kernel_size=downscale, stride=downscale)
        self.conv = nn.Conv2d(c_lgan, c_lgan, kernel_size=3, stride=1, padding=1)
        self.norm = Norm(c_lgan)
        self.act = ACT()
        self.up = nn.ConvTranspose2d(c_lgan, c_lgan, kernel_size=downscale, stride=downscale)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.up(x)
        return x


class Norm(nn.Module):
    def __init__(self, c_in):
        super(Norm, self).__init__()
        #     self.norm = nn.BatchNorm2d(c_in)
        self.norm = nn.GroupNorm(4, c_in, eps=1e-6, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

    # def __init__(self, c_in):
    #     super(Norm, self).__init__()
    #     self.norm = nn.LayerNorm(c_in, elementwise_affine=False)
    #
    # def forward(self, x):
    #     b, c, h, w = x.shape
    #     x = rearrange(x, 'b c h w->b (h w) c')
    #     x = self.norm(x)
    #     x = rearrange(x, 'b (h w) c->b c h w', w=w, h=h)
    #     return x


class Head(nn.Module):
    def __init__(self, c_lgan, c_in, down_sample):
        super(Head, self).__init__()
        self.conv1 = ShiftConv2d(c_in, c_lgan)
        self.norm = Norm(c_lgan)
        self.act = ACT()
        self.conv2 = ShiftConv2d(c_lgan, c_lgan)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class Tail(nn.Module):
    def __init__(self, c_lgan, c_in, down_sample):
        super(Tail, self).__init__()
        self.conv1 = nn.Conv2d(c_lgan, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 4, kernel_size=3, stride=1, padding=1)
        # self.norm = nn.BatchNorm2d(c_in * down_sample * down_sample)
        # self.norm = Norm(c_in * down_sample * down_sample)
        # self.act = act
        # self.conv2 = nn.Conv2d(c_in * down_sample * down_sample, c_in * down_sample * down_sample, kernel_size=3,
        #                        stride=1, padding=1)
        # self.ps = nn.PixelShuffle(down_sample)

    def forward(self, x):
        x = self.conv1(x)
        # shortcut = x
        # x = self.norm(x)
        # x = self.act(x)
        # x = self.conv2(x) + shortcut
        x = self.conv2(x)

        x = self.conv3(x)
        return x


# class Tail(nn.Module):
#     def __init__(self, c_lgan, c_in, down_sample):
#         super(Tail, self).__init__()
#         self.c_in = c_in
#         self.down_sample = down_sample
#         self.fc = nn.Linear(c_lgan, c_in * down_sample * down_sample)
#         self.norm = nn.LayerNorm(c_lgan)

# def forward(self, x):
#     b, c, h, w = x.shape
#     x = rearrange(x, 'b c h w->b (h w) c')
#     x = self.norm(x)
#     x = self.fc(x)
#     x = rearrange(x, 'b (h w) (c d1 d2)->b c (d1 h) (d2 w)',
#                   c=self.c_in, d1=self.down_sample, d2=self.down_sample, h=h, w=w)
#     x = nn.functional.interpolate(x, size=[721, 1440], mode='bilinear')
#     return x


class FD(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4):
        super(FD, self).__init__()
        # self.fc1 = MLP(inp_channels, inp_channels * exp_ratio)
        # self.fc2 = MLP(inp_channels * exp_ratio, out_channels)
        self.fc1 = ShiftConv2d(inp_channels, inp_channels * exp_ratio)
        self.fc2 = ShiftConv2d(inp_channels * exp_ratio, out_channels)
        self.act1 = ACT()
        self.act2 = ACT()

    def forward(self, x):
        y = self.fc1(x)
        y = self.act1(y)
        y = self.fc2(y)

        return y


class GeoAB(nn.Module):

    def __init__(self, channels, window_size=5, num_heads=8, split_part=3):
        super(GeoAB, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.f_split_chns = [int(channels / split_part) for _ in range(split_part)]  # f_split_chns = [72, 72, 72]
        self.split_chns = [int(channels * 2 / split_part) for _ in range(split_part)]  # split_chns = [144, 144, 144]
        # self.project_inp = MLP(channels, channels * 3)
        # self.project_out = MLP(channels, channels)
        self.f_project_inp = ShiftConv2d(channels, channels)
        self.project_inp = ShiftConv2d(channels, channels * 2)
        self.project_out = ShiftConv2d(channels, channels)
        # self.f_project_inp = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        # self.project_inp = nn.Conv2d(channels, channels * 2, kernel_size=1, stride=1)
        # self.project_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)
        self.lr_logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)
        # #########################################################################
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, self.num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)

        #  x,y = torch.meshgrid(a,b,indexing='ij')把数组a 当做一列，再根据b的长度扩充列，当数组 b 当做一行，再根据a 的长度扩充行，返回的是二维张量，分别表示网格中的横纵坐标
        # torch.stack 将 torch.meshgrid 生成的两个张量x,y拼接在一起，permute 进行维度转换，contiguous 将张量转为连续存储，unsqueeze 升一个维度
        # contigous()确保张量是内存连续布局
        relative_coords_table = (torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij'))
        .permute(1, 2, 0).contiguous().unsqueeze(
            0))  # (1,2*Window_size-1,2*Window_size-1,2)  (1, 9, 9, 2)

        relative_coords_table[:, :, :, 0] /= (self.window_size - 1)  # (1, 9, 9) normalize to -1,1
        relative_coords_table[:, :, :, 1] /= (self.window_size - 1)  # (1, 9, 9) normalize to -1,1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        # torch.sign进行符号函数操作，torch.abs做绝对值操作
        # 不需要学习的参数
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 两个向量相减，利用该广播机制，(2,wh*wh,1)-(2,1,wh*wh)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # 注册relative_position_index为不需要学习的参数
        # #########################################################################

    def wa(self, x, wsize):
        b, c, h, w = x.shape
        k, v = rearrange(
            x, 'b (kv head c) (h dh) (w dw) -> kv (b h w) head (dh dw) c',
            kv=2, dh=wsize, dw=wsize, head=self.num_heads
        )
        # @等同于torch.matmul，转置后就变成了((b h0 w0), 8, 25, 9)@((b h0 w0), 8, 9, 25)->((b h0 w0), 8, 25, 25)
        atn = (F.normalize(k, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        t = torch.tensor(1. / 0.01).to('cuda')
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(t)).exp()
        atn = atn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1,
                                                                                     self.num_heads)  # (1, 9, 9, 2)->(1, 9, 9, 8)->(81, 8)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        atn = atn + relative_position_bias.unsqueeze(0)
        ###
        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(y_, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
                       h=h // wsize, w=w // wsize, dh=wsize, dw=wsize, head=self.num_heads)
        return y_

    # def forward(self, x, roll=False):
    #     y_ = self.project_inp(x)
    #     _, _, h, w = y_.shape
    #     wsize = self.window_size
    #     shifted window attention
    # if roll:
    #     y_ = torch.roll(y_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
    # y_ = self.wa(y_, wsize, (h, w))
    # if roll:
    #     y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
    # y = self.project_out(y_)
    # return y
    def forward(self, x, roll=False):
        b, c, h, w = x.shape
        x = self.project_inp(x)  # b 2*216 h w
        # xs for key and value
        xs = torch.split(x, self.split_chns, dim=1)  # 3个 (b, 144, h, w)
        wsize = self.window_size
        ys = []

        # window attention
        y_ = self.wa(xs[0], wsize)
        ys.append(y_)

        # shifted window attention
        x_ = torch.roll(xs[1], shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
        y_ = self.wa(x_, wsize)
        y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
        ys.append(y_)

        # long-range attentin
        # for longitude
        k, v = rearrange(xs[2], 'b (kv head c) h w -> kv (b h) head w c', kv=2, head=self.num_heads)
        atn = (F.normalize(k, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        t = torch.tensor(1. / 0.01).to('cuda')
        logit_scale = torch.clamp(self.lr_logit_scale, max=torch.log(t)).exp()
        atn = atn * logit_scale
        # atn = (q @ k.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # y_ = rearrange(v, '(b h) w c-> b c h w', b=b)
        # ys.append(y_)

        # for latitude
        k, v = rearrange(k, '(b h) head w c -> (b w) head h c', h=h), rearrange(v, '(b h) head w c -> (b w) head h c',
                                                                                h=h)
        atn = (F.normalize(k, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        atn = atn * logit_scale
        # atn = (q @ k.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        y_ = rearrange(v, '(b w) head h c-> b (head c) h w', b=b)
        ys.append(y_)

        y = torch.cat(ys, dim=1)
        y = self.project_out(y)

        return y


class FEB(nn.Module):
    def __init__(self, inp_channels, exp_ratio=2, window_size=5, num_heads=8, downscale=2):
        super(FEB, self).__init__()
        self.exp_ratio = exp_ratio
        self.inp_channels = inp_channels
        self.downscale = downscale

        # self.down = DownBlock(inp_channels, downscale=downscale)  # c*downscale
        # self.up = UpBlock(inp_channels, downscale=downscale)

        # self.FD = FD(inp_channels=inp_channels * downscale, out_channels=inp_channels * downscale, exp_ratio=exp_ratio)
        # self.ATT = GeoAB(channels=inp_channels * downscale, window_size=window_size, num_heads=num_heads)
        # self.norm1 = Norm(inp_channels * downscale)
        # self.drop = nn.Dropout2d(0.2)
        # self.norm2 = Norm(inp_channels * downscale)
        self.FD = FD(inp_channels=inp_channels, out_channels=inp_channels, exp_ratio=exp_ratio)
        self.ATT = GeoAB(channels=inp_channels, window_size=window_size, num_heads=num_heads)
        self.norm1 = Norm(inp_channels)
        self.drop = nn.Dropout2d(0.2)
        self.norm2 = Norm(inp_channels)

    def forward(self, x, roll=False):
        res = x
        # x = self.down(x)  # c:c * downscale
        shortcut = x
        x = self.ATT(x, roll)
        x = self.drop(x)
        x = self.norm1(x) + shortcut

        shortcut = x
        x = self.FD(x)
        x = self.norm2(x) + shortcut
        # x = self.up(x)
        x = x + res
        return x


if __name__ == '__main__':
    A = torch.randn(2, 3, 4, 5)
    B = torch.randn(2, 3, 5, 8)
    # result = torch.matmul(A, B)
    # C = A@B
    # print(C.shape)
    fa = torch.nn.Linear(5, 512, bias=True)
    C = fa(A)
    print(C.shape)
