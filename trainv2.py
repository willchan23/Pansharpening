import torch
import math

import yaml

from metrics.metrics2 import cpsnr, cssim, scale_to_255
from util import utils
import os
import sys
import random

import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataset.datasetV2 import Benchmark
import numpy as np

from metrics.latitude_weighted_loss import LatitudeLoss
from tqdm import tqdm
from jacksung.utils.log import LogClass, oprint
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from datetime import datetime
import jacksung.utils.fastnumpy as fnp
from jacksung.utils.log import StdLog

from util.cache import Cache
from util.cache_loader import CacheLoader
from util.data_parallelV2 import BalancedDataParallel
from util.norm_util import Normalization
from einops import rearrange
from util.utils import EXCLUDE_DATE

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

    device, args = utils.parse_config()
    # definitions of model

    model = utils.get_model(args)
    model = model.to(device)
    # load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load(ckpt['model_state_dict'], strict=False)
    # definition of loss and optimizer
    loss_func = eval(args.loss)
    if args.fp == 16:
        eps = 1e-3
    elif args.fp == 64:
        eps = 1e-13
    else:
        eps = 1e-8
    optimizer = eval(f'torch.optim.{args.optimizer}(model.parameters(), lr=args.lr, eps=eps)')
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)
    # resume training
    if args.resume is not None:
        ckpt_files = os.path.join(args.resume, 'models', "model_latest.pt")
        if len(ckpt_files) != 0:
            ckpt = torch.load(ckpt_files)
            prev_epoch = ckpt['epoch']
            start_epoch = prev_epoch + 1
            model.load(ckpt['model_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            # reset folder and param
            experiment_path = args.resume
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('Select {} file, resume training from epoch {}.'.format(ckpt_files, start_epoch))
        else:
            raise Exception(f'{os.path.join(args.resume, "models", "model_latest.pt")}中无有效的ckpt_files')
    else:
        start_epoch = 1
        # auto-generate the output log name
        experiment_name = None
        timestamp = cur_timestamp_str()
        experiment_name = '{}-{}'.format(args.model if args.log_name is None else args.log_name, timestamp)
        # 处理日志
        experiment_path = os.path.join(args.log_path, experiment_name)

        # float('inf')：指标的默认值为正无穷大，float('0')：指标的默认值为0.0
        stat_dict = utils.get_stat_dict(
            (
                ('val-loss', float('inf'), '<'),
                ('PSNR', float('0'), '>'),
                ('SSIM', float('0'), '>'),
            )
        )
        # create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        # save training parameters
        # 使用 vars(args) 将 args 对象的属性转换为字典，然后保存备份
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config_saved.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            # yaml.dump() function write dict to yaml format
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)
        # 初始化神经网络模型的权重
        model.init_model()
    # 将模型移动到指定设备上
    model = model.to(device)
    if args.balanced_gpu0 >= 0:
        # balance multi gpus
        model = BalancedDataParallel(args.balanced_gpu0, model, device_ids=list(range(len(args.gpu_ids))))
    else:
        # multi gpus
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_ids))))
    log_name = os.path.join(experiment_path, 'log.txt')
    warning_path = os.path.join(experiment_path, 'warning.txt')
    stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
    # 将程序中所有的标准输出（包括 print() 函数输出的内容）都发送到 StdLog 类的实例中，然后根据 StdLog 类中定义的逻辑来处理这些输出消息
    sys.stdout = StdLog(filename=log_name, common_path=warning_path)

    # 循环计算了模型中每个参数的参数数量，并打印出总参数数量以及数据路径
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    # 首先计算 num_params / 1024 ** 2，即将参数数量转换为以兆（M）为单位的值，然后使用 round() 函数保留两位小数，并将结果转换为字符串
    print('Total Number of Parameters:' + str(round(num_params / 1024 ** 2, 2)) + 'M')
    print('loading train data: ')
    train_dataset = Benchmark(args.train_data_path, train=True, repeat=args.repeat)
    print('loading valid data: ')
    valid_dataset = Benchmark(args.valid_data_path, train=False, repeat=args.repeat)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                  shuffle=False, pin_memory=False, drop_last=False)
    # start training
    # 计时器（Stopwatch），用于测量代码执行的时间
    sw = Stopwatch()
    rt = RemainTime(args.epochs)
    cloudLogName = experiment_path.split(os.sep)[-1]
    log = LogClass(args.cloudlog == 'on')
    log.send_log('Start training', cloudLogName)
    log_every = max(len(train_dataloader) // args.log_lines, 1)

    # train_norm = Normalization(args.train_data_path)
    # train_norm.input_mean, train_norm.input_std, train_norm.gt_mean, train_norm.gt_std = utils.data_to_device(
    #     [train_norm.input_mean, train_norm.input_std, train_norm.gt_mean, train_norm.gt_std], device, args.fp)
    #
    # valid_norm = Normalization(args.valid_data_path)
    # valid_norm.input_mean, valid_norm.input_std, valid_norm.gt_mean, valid_norm.gt_std = utils.data_to_device(
    #     [valid_norm.input_mean, valid_norm.input_std, valid_norm.gt_mean, valid_norm.gt_std], device, args.fp)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        # lr :learning rate
        opt_lr = scheduler.get_last_lr()
        print('##===============-fp{}- Epoch: {}, lr: {} =================##'.format(args.fp, epoch, opt_lr))
        # 多线程
        train_dataloader.check_worker_number_rationality()
        # training the model
        for iter_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            ms, pan, gt = utils.data_to_device(batch, device, args.fp)
            # input_norm, gt_norm = train_norm.input_norm(input), train_norm.gt_norm(gt)

            roll = 0
            y_ = model(pan, ms, roll)
            b, c, h, w = y_.shape
            loss = loss_func(y_, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            # print log of the loss
            if (iter_idx + 1) % log_every == 0:
                cur_steps = (iter_idx + 1) * args.batch_size
                total_steps = len(train_dataloader) * args.batch_size
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)
                ## 在输出时保持数字的对齐
                # 计算 args.epochs 的数量级，并将其向上取整为最接近的整数
                epoch_width = math.ceil(math.log10(args.epochs))
                # str.zfill() 方法会在字符串左侧填充0，使得字符串达到指定的宽度
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter_idx + 1)
                stat_dict['losses'].append(avg_loss)

                oprint('Epoch:{}, {}/{}, Loss: {:.4f}, T:{}'.format(
                    cur_epoch, cur_steps, total_steps, avg_loss, sw.reset()))
        # validating the model
        if epoch % args.test_every == 0:
            torch.set_grad_enabled(False)
            model = model.eval()
            epoch_loss = 0
            psnr_list, ssim_list, qnr_list, d_lambda_list, d_s_list = [], [], [], [], []
            progress_bar = tqdm(total=len(valid_dataset), desc='Infer')
            count = 0
            for iter_idx, batch in enumerate(valid_dataloader):
                optimizer.zero_grad()
                ms, pan, gt = utils.data_to_device(batch, device, args.fp)
                # input_norm, gt_norm = valid_norm.input_norm(input), valid_norm.gt_norm(gt)

                roll = 0
                y_ = model(pan, ms, roll)
                b, c, h, w = y_.shape

                # quantize output to [0, 255]

                loss = loss_func(y_, gt)
                # y_ = valid_norm.denorm(y_)
                y_ = y_.clamp(0, 1)
                gt = gt.clamp(0, 1)
                batch_psnr, batch_ssim, batch_qnr, batch_D_lambda, batch_D_s = [], [], [], [], []
                for batch_index in range(b):
                    # 计算PSNR和SSIM
                    predict_y = (y_[batch_index, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    ground_truth = (gt[batch_index, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    psnr = cpsnr(predict_y, ground_truth)
                    ssim = cssim(predict_y, ground_truth, 255)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)

                epoch_loss += float(loss)
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                count += 1
                progress_bar.update(b)
            progress_bar.close()
            epoch_loss = epoch_loss / count
            psnr_total = np.array(psnr_list).mean()
            ssim_total = np.array(ssim_list).mean()
            # rmse = rmse / count
            # rr = rr / count

            log_out = utils.make_best_metric(stat_dict,
                                             (
                                                 ('val-loss', float(epoch_loss)),
                                                 ('PSNR', psnr_total),
                                                 ('SSIM', ssim_total)
                                             ),
                                             epoch, (experiment_model_path, model, optimizer, scheduler),
                                             (log, args.epochs, cloudLogName))
            # print the log & flush out
            print(log_out)
            # save the stat dict
            # save training parameters
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
            torch.set_grad_enabled(True)
            model = model.train()
        # update scheduler
        scheduler.step()
        rt.update()
    log.send_log('Training Finished!', cloudLogName)
    utils.draw_lines(stat_dict_name)
