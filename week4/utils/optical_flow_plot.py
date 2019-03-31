import os

import numpy as np
import cv2

import matplotlib.pyplot as plt


def plot_opticalflow_gt(opticalflow, sequence, step=10, title='', save_path='plots/', need_conversion=False):
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
        print(maxOF)

        x, y = np.meshgrid(np.arange(0, w*step, step), np.arange(0, h*step, step))

        flow_color = flow_to_color(flow)

        plt.imshow(flow_color)
        plt.title(title)
        plt.savefig(save_path + title[:2] + '_colorwheel' + str(ind) + '.png')
        plt.show()
        plt.close()

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


def plot_opticalflow_bm(flow, sequence, step=10, title='', save_path='../plots/'):
    #fu = (flow[:, :, 0] - 2. ** 15) / 64
    #fv = (flow[:, :, 1] - 2. ** 15) / 64
    #flow = np.transpose(np.array([fu, fv]))
    U = np.transpose(flow[:, :, 0])
    V = np.transpose(flow[:, :, 1])

    w, h = flow.shape[:2]

    maxOF = max(np.max(U), np.max(V))
    print(maxOF)

    x, y = np.meshgrid(np.arange(0, w*step, step), np.arange(0, h*step, step))

    flow_color = flow_to_color(flow)

    plt.imshow(flow_color)
    plt.title(title)
    plt.savefig(save_path + title[:2] + '_colorwheel' + '.png')
    plt.show()
    plt.close()

    plt.imshow(sequence[0])
    M = np.hypot(U, V)
    plt.quiver(x, y, U, -V, M, scale=maxOF*20, alpha=1, width=0.005)
    plt.title(title)
    plt.savefig(save_path + title[:2] + '_colorimage' + '.png')
    plt.show()
    plt.close()

    plt.imshow(sequence[0])
    plt.quiver(x, y, U, -V, scale=maxOF*10, alpha=1, color='r')
    plt.title(title)
    plt.savefig(save_path + title[:2] + '_redimage' + '.png')
    plt.show()
    plt.close()


def read_opticalflow(path):
    imgNames = os.listdir(path)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('000045_10.png'):
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
        if name.endswith('000045_10.png') or name.endswith('000045_11.png'):
            im = cv2.imread(path + name)
            seq_images.append(im)
    return seq_images

# Next functions have been obtained from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    #assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    #assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = np.transpose(flow_uv[:,:,0])
    v = np.transpose(flow_uv[:,:,1])

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)



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