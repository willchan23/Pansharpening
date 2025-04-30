import os

import numpy as np
import matplotlib.pyplot as plt


def npy_to_rgb_custom(npy, output_path):
    # Load the .npy file
    if npy.endswith('.npy'):
        data = np.load(npy)  # Shape: (4, h, w)
    else:
        data = npy
    # Extract R, G, B channels
    r = data[2]  # 3rd channel
    g = data[1]  # 2nd channel
    b = data[0]  # 1st channel

    # Normalize each channel to 0-1 range using min-max normalization
    def normalize(channel, index):
        lms = np.load(r'E:\pycode\Pansharpening\data\GF2\Test\origion_lms\lms.npy')[0, index, ...]
        lms_max = lms.max()
        lms_min = lms.min()
        channel = channel.clip(lms_min, lms_max)
        return (channel - lms_min) / (lms_max - lms_min)

    r = normalize(r, 2)
    g = normalize(g, 1)
    b = normalize(b, 0)

    # Stack channels into an RGB image
    rgb = np.stack((r, g, b), axis=-1)  # Shape: (h, w, 3)

    # Display and save the image
    plt.imshow(rgb)
    plt.axis('off')  # Remove axes
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def npy_to_rgb_custom1(npy, output_path, index):
    # Load the .npy file
    # if npy.endswith('.npy'):
    #     data = np.load(npy)  # Shape: (4, h, w)
    # else:
    data = npy
    # Extract R, G, B channels
    r = data[2]  # 3rd channel
    g = data[1]  # 2nd channel
    b = data[0]  # 1st channel

    # Normalize each channel to 0-1 range using min-max normalization
    def normalize(channel):
        min_val = np.min(channel)
        max_val = np.max(channel)
        return (channel - min_val) / (max_val - min_val) if max_val > min_val else channel

    r = normalize(r)
    g = normalize(g)
    b = normalize(b)

    # Stack channels into an RGB image
    rgb = np.stack((r, g, b), axis=-1)  # Shape: (h, w, 3)

    # Display and save the image
    plt.imshow(rgb)
    plt.axis('off')  # Remove axes
    plt.savefig(output_path + index + '.png', bbox_inches='tight', pad_inches=0)
    plt.show()


# Example usage
if __name__ == '__main__':
    # folder_path = "E:\pycode\Pansharpening\data\GF2\Test\proc\geoan-2025-0427-1605-5907"
    # for file in os.listdir(folder_path):
    #     if file.endswith('.npy'):
    #         file_path = os.path.join(folder_path, file)
    #         npy_to_rgb_custom(file_path,
    #                           "E:\pycode\Pansharpening\\fig\Test用lms的均值标准差一次denorm\\" + file.replace('.npy',
    # '.png'))
    # file_path = "E:\pycode\Pansharpening\data\GF2\Test\lms.npy"
    # npy = np.load(file_path)
    # for index in range(0, 20):
    #     npy_to_rgb_custom1(npy[index, ...], "E:\pycode\Pansharpening\data\GF2\Test\lms", str(index))
    file_path = r"E:\pycode\Pansharpening\data\GF2\Test\proc\geoan-2025-0427-1605-5907\0-iters.npy"
    npy_to_rgb_custom(file_path, file_path.replace('.npy', '.png'))
    # npy1 = np.load(r"E:\pycode\Pansharpening\data\GF2\Test\proc\geoan-2025-0427-1605-5907\0-iters.npy")
    # npy2 = np.load(r"E:\pycode\Pansharpening\data\GF2\Test\proc\geoan-2025-0427-1605-5907\0-iters_2std.npy")
    # pass
