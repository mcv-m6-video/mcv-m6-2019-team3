import os

import numpy as np
import cv2

import matplotlib.pyplot as plt


def plot_opticalflow(opticalflow, sequence, step=10, title='', save_path='../plots/', need_conversion=False):

    for ind, image in enumerate(opticalflow):
        image = cv2.resize(image, (0, 0), fx=1./step, fy=1./step)
        if need_conversion:
            flow = convert_flow_data(image)
            valid = np.transpose(flow[:, :, 2])
        U = np.transpose(flow[:, :, 0])
        V = np.transpose(flow[:, :, 1])

        w, h = flow.shape[:2]
        if need_conversion:
            U = U*valid
            V = V*valid

        maxOF = max(np.max(U), np.max(V))

        x, y = np.meshgrid(np.arange(0, w*step, step), np.arange(0, h*step, step))

        plt.imshow(sequence[ind])
        M = np.hypot(U, V)
        plt.quiver(x, y, U, -V, M, scale=maxOF*20, alpha=1, width=0.005)
        plt.title(title)
        plt.savefig(save_path + title[:2] + '_colorimage' + str(ind) + '.png')
        plt.show()
        plt.close()

        plt.imshow(sequence[ind])
        plt.quiver(x, y, U, -V, scale=maxOF*10, alpha=1, color='r')
        plt.title(title)
        plt.savefig(save_path + title[:2] + '_redimage' + str(ind) + '.png')
        plt.show()
        plt.close()


def read_opticalflow(path):
    imgNames = os.listdir(path)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('000045_10.png') or name.endswith('000157_10.png'):
            images.append(cv2.imread(path + name, -1))
    return images


def convert_flow_data(img):

    fu = (img[:, :, 2] - 2. ** 15) / 64
    fv = (img[:, :, 1] - 2. ** 15) / 64
    valid = img[:, :, 0]

    flow_test = np.transpose(np.array([fu, fv, valid]))

    return flow_test


def read_sequences(path):
    imgNames = os.listdir(path)
    imgNames.sort()
    seq_images = []
    for name in imgNames:
        if name.endswith('000045_10.png') or name.endswith('000157_11.png'):
            im = cv2.imread(path + name)
            seq_images.append(im)
    return seq_images


if __name__ == '__main__':
    opticalflow_path = '../datasets/results_opticalflow_kitti/results/'
    sequences_path = '../datasets/data_stereo_flow/training/image_0/'
    gt_path = '../datasets/data_stereo_flow/training/flow_noc/'
    save_path = '../plots/'


    opticalflow_images = read_opticalflow(opticalflow_path)
    gt_opticalflow = read_opticalflow(gt_path)
    first_sequence = read_sequences(sequences_path)

    plot_opticalflow(opticalflow_images, first_sequence, title='KL detections', save_path=save_path+'KL_optflow/')
    plot_opticalflow(gt_opticalflow, first_sequence, title='GT optical flow', save_path=save_path+'GT_optflow/')