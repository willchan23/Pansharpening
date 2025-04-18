import os
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta

# from dateutil.relativedelta import relativedelta

basePath = r'E:\pycode\Pansharpening\data\GF2\Valid'


def cldas_mean_and_std():
    for year in ['2020']:
        yearPath = basePath + year
        monthList = os.listdir(yearPath)
        for month in tqdm(monthList, desc='Processing month'):
            if month == '04_old':
                continue
            sumArray = []
            monthPath = os.path.join(yearPath, month)
            dayList = os.listdir(monthPath)
            for day in dayList:
                dayPath = os.path.join(monthPath, day)
                fileList = os.listdir(dayPath)
                for file in fileList:
                    filePath = os.path.join(dayPath, file)
                    data = np.load(filePath)
                    sumArray.append(data)
            sumArray = np.array(sumArray)
            meanArray = np.mean(sumArray, axis=0)
            stdArray = np.std(sumArray, axis=0)
            totalMean = np.mean(meanArray, axis=(1, 2))
            totalStd = np.std(stdArray, axis=(1, 2))
            savePath = basePath + f'{year}/{month}/'
            if not os.path.exists(savePath):
                # 如果目录不存在，则创建目录
                os.makedirs(savePath)
            np.save(savePath + f'/{year}{month}-pixel-mean.npy', meanArray)
            np.save(savePath + f'/{year}{month}-pixel-std.npy', stdArray)
            np.save(savePath + f'/{year}{month}-total-mean.npy', totalMean)
            np.save(savePath + f'/{year}{month}-total-std.npy', totalStd)
            print(f'finish the  {year}-{month}')


def pansharpening_mean_and_std(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 遍历所有文件
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        # data 为 (b,c,h,w)的 计算得到均值和方差维度为(c)
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        # 保存均值和方差
        np.save(file_path.replace('.npy', '-mean.npy'), mean)
        np.save(file_path.replace('.npy', '-std.npy'), std)


if __name__ == '__main__':
    pansharpening_mean_and_std(r'/mnt/data1/czx/Pansharpening/GF2/Train')
