import os

import numpy as np
import cv2

import matplotlib.pyplot as plt


def plot_opticalflow(opticalflow, sequence, step=10):

    for ind, image in enumerate(opticalflow):
        image = cv2.resize(image, (0, 0), fx=1./step, fy=1./step)
        flow = convert_flow_data(image)
        valid = np.transpose(flow[:, :, 2] == 1)
        U = np.flip(np.transpose(flow[:, :, 0]), axis=0)
        V = np.flip(np.transpose(flow[:, :, 1]), axis=0)

        w, h = flow.shape[:2]
        U = np.reshape(U[valid], (h, w))
        V = np.reshape(V[valid], (h, w))

        maxOF = max(np.max(U), np.max(V))

        x, y = np.meshgrid(np.arange(0, w*step, step), np.arange(0, h*step, step))
        x = np.reshape(x[valid], (h, w))
        y = np.reshape(y[valid], (h, w))
        plt.imshow(sequence[ind])

        plt.quiver(x, y, U, V, scale=maxOF*20, alpha=1, color='r')
        plt.title('LK Optical Flow Results')
        plt.savefig('OpticalFlow_image' + str(ind) + '.png')
        plt.show()
        plt.close()


def read_opticalflow(path):
    imgNames = os.listdir(path)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('.png'):
            images.append(cv2.imread(path + name, -1))
    return images


def convert_flow_data(img_test):

    fu_test = (img_test[:, :, 2] - 2. ** 15) / 64
    fv_test = (img_test[:, :, 1] - 2. ** 15) / 64
    valid_test = img_test[:, :, 0]

    flow_test = np.transpose(np.array([fu_test, fv_test, valid_test]))

    return flow_test


def read_sequences(path):
    imgNames = os.listdir(path)
    imgNames.sort()
    seq_images = []
    for name in imgNames:
        if name.endswith('.png'):
            if (name[:9] == '000045_10') or (name[:9] == '000157_10') :
                im = cv2.imread(path + name)
                seq_images.append(im)
    return seq_images


if __name__ == '__main__':
    opticalflow_path = '../datasets/results_opticalflow_kitti/results/'
    sequences_path = '../datasets/data_stereo_flow/training/image_0/'

    opticalflow_images = read_opticalflow(opticalflow_path)
    first_sequence = read_sequences(sequences_path)

    plot_opticalflow(opticalflow_images, first_sequence)