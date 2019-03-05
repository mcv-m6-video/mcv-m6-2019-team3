import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import os
import cv2


def OFplots(ofImages, images):
    step = 10
    ind = 0

    for ofIm in ofImages:
        ofIm = cv2.resize(ofIm, (0, 0), fx=1. / step, fy=1. / step)
        rows, cols, depth = ofIm.shape
        U = []
        V = []

        for pixel in range(0, ofIm[:, :, 0].size):
            isOF = ofIm[:, :, 0].flat[pixel]
            if isOF == 1:
                U.append((((float)(ofIm[:, :, 1].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
                V.append((((float)(ofIm[:, :, 2].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
            else:
                U.append(0)
                V.append(0)
        print(np.max(U))
        print(U[:30])
        U = np.reshape(U, (rows, cols))
        V = np.reshape(V, (rows, cols))
        print(U.shape)
        x, y = np.meshgrid(np.arange(0, cols * step, step), np.arange(0, rows * step, step))

        plt.imshow(images[ind])
        plt.quiver(x, y, U, V, scale=0.1, alpha=1, color='r')
        plt.title('Optical Flow')
        plt.savefig('OF' + str(ind) + '.png')
        plt.show()
        plt.close()
        ind += 1


def plot_opticalflow(opticalflow, sequence, step=10):

    for ind, image in enumerate(opticalflow):
        image = cv2.resize(image, (0, 0), fx=1./step, fy=1./step)
        flow = convert_flow_data(image)
        valid = np.transpose(flow[:, :, 2] == 1)
        U = np.transpose(flow[:, :, 0]/200)
        V = np.transpose(flow[:, :, 1]/200)

        w, h = flow.shape[:2]
        U = np.reshape(U[valid], (h, w))
        V = np.reshape(V[valid], (h, w))
        x, y = np.meshgrid(np.arange(0, w*step, step), np.arange(0, h*step, step))
        x = np.reshape(x[valid], (h, w))
        y = np.reshape(y[valid], (h, w))
        plt.imshow(sequence[ind])
        plt.quiver(x, y, U, -V, scale=0.1, alpha=1, color='r')
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