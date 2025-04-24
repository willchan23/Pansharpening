import rasterio
from jacksung.utils.data_convert import np2tif
import h5py
import numpy as np
from rasterio.transform import from_origin


def h52npy(h5_file_path, output_dir):
    """
        将 HDF5 文件中的所有数据集转换为单独的 .npy 文件
        参数：
            h5_file_path: 输入的 HDF5 文件路径
            output_dir: 输出目录（默认当前目录）
        """
    with h5py.File(h5_file_path, 'r') as f:
        # 递归遍历 HDF5 文件中的所有对象
        def visit_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 从数据集中读取数据为 NumPy 数组
                data = obj[:]
                # 生成输出文件名（替换路径分隔符为下划线）
                npy_filename = f"{output_dir}/{name.replace('/', '_')}.npy"
                # 保存为 .npy 文件
                np.save(npy_filename, data)
                print(f"已保存: {npy_filename}")

        # 遍历所有组和数据集
        f.visititems(visit_dataset)


def save_3d_to_tif(np_data, save_path, left=None, top=None, x_res=None, y_res=None, dtype=None):
    """
    Save a (3, h, w) NumPy array as a multi-band GeoTIFF file.

    :param np_data: Input NumPy array with shape (3, h, w)
    :param save_path: Path to save the output TIFF file
    :param left: Left coordinate of the top-left corner (optional)
    :param top: Top coordinate of the top-left corner (optional)
    :param x_res: Resolution in the x-direction (optional)
    :param y_res: Resolution in the y-direction (optional)
    :param dtype: Data type for the output file (optional)
    """
    if np_data.shape[0] != 3:
        raise ValueError("Input array must have shape (3, h, w)")

    _, h, w = np_data.shape
    transform = None

    if left is not None and top is not None and x_res is not None and y_res is not None:
        transform = from_origin(left, top, x_res, y_res)

    with rasterio.open(
            save_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,  # Number of bands
            dtype=dtype if dtype else np_data.dtype,
            crs="EPSG:4326" if transform else None,
            transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(np_data[i, :, :], i + 1)

    print(f"TIFF file saved at: {save_path}")


# def npy2tif(npyPath)
def npy2Tiff(npyPath, index):
    """
    将npy文件转换为RGB图像
    :param npyPath: npy文件路径
    :return:
    """
    data = np.load(npyPath)
    # 输入一个4D数组，形状为 (200, 4 , h, w)
    # 选择前3个波段
    data = data[index, :3, :, :]
    save_3d_to_tif(data, npyPath.replace('.npy', '.tif'), dtype=data.dtype)


def max_min_to_01(npy_path):
    npy = np.load(npy_path)
    max_val = npy.max(axis=(2, 3), keepdims=True)
    min_val = npy.min(axis=(2, 3), keepdims=True)
    # return (npy - min_val) / (max_val - min_val + 1e-8)
    npy = (npy - min_val) / (max_val - min_val + 1e-8)
    save_path = npy_path.replace('.npy', '_01.npy')
    np.save(save_path, npy)


if __name__ == '__main__':
    # h52npy(r"/mnt/data1/czx/Pansharpening/GF2/train_gf2.h5")
    # h52npy(r"E:\data\Pansharpening\training_gf2\valid_gf2.h5")
    h52npy(r'/mnt/data1/czx/Pansharpening/GF2/train_gf2.h5', '/mnt/data1/czx/Pansharpening/GF2v2/Train')
    h52npy(r'/mnt/data1/czx/Pansharpening/GF2/valid_gf2.h5', '/mnt/data1/czx/Pansharpening/GF2v2/Valid')
    # npy = np.load(r"E:\pycode\Pansharpening\util\lms.npy")
    # t = np2tif()
    # npy2Tiff(r"E:\pycode\Pansharpening\util\gt.npy", 0)
    npy_list = ['ms.npy', 'pan.npy', 'gt.npy']
    for npy in npy_list:
        max_min_to_01(rf"/mnt/data1/czx/Pansharpening/GF2v2/Train/{npy}")
        max_min_to_01(rf"/mnt/data1/czx/Pansharpening/GF2v2/Valid/{npy}")
