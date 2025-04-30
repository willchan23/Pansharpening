import numpy as np


def standardize_4chw_npy(npy_path, output_path):
    # Load the .npy file
    data = np.load(npy_path)  # Shape: (4, h, w)

    # Standardize each channel using its own mean and std
    def standardize(channel):
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        return (channel - mean_val) / std_val
        # Apply standardization to each channel

    standardized_data = standardize(data)

    ms_npy = np.load(r"E:\pycode\Pansharpening\data\GF2\Test\origion_lms\ms.npy")[0, ...]
    # 再用ms的均值和标准差来denorm
    ms_mean = np.mean(ms_npy)
    ms_std = np.std(ms_npy)
    data = standardized_data * ms_std + ms_mean
    np.save(output_path, data)


# Example usage
npy_path = r"E:\pycode\Pansharpening\data\GF2\Test\proc\geoan-2025-0427-1605-5907\0-iters.npy"
output_path = npy_path.replace('.npy', '_2std.npy')
standardize_4chw_npy(npy_path, output_path)
# if __name__ == '__main__':
#     # 生成一个测试
#     test_data = np.random.rand(4, 100, 100)  # Example data
#     test_mean = np.mean(test_data)
#     test_std = np.std(test_data)
#     # norm
#     standardized_test_data = (test_data - test_mean) / test_std
#     # denorm
#     out = standardized_test_data * test_std + test_mean
#     pass
if __name__ == '__main__':
    file_path = r"E:\pycode\Pansharpening\data\GF2\Test\proc\geoan-2025-0427-1605-5907\0-iters_2std.npy"
