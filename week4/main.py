import os
from itertools import product
import numpy as np

from block_matching import block_matching_optical_flow
from optical_flow import pyflow, farneback, lucas_kanade
from utils.optical_flow_metrics import compute_msen, read_flow_data
from utils.optical_flow_plot import plot_opticalflow_bm, plot_opticalflow_gt, read_opticalflow, read_sequences
from utils.plotting import create_folder
from video_stabilizer import video_stabilization, point_feature_matching

sequences_path = '../datasets/data_stereo_flow/training/image_0/'

gt_path = '../datasets/data_stereo_flow/training/flow_noc/'
gt_noc = "../datasets/data_stereo_flow/training/flow_noc/000045_10.png"

opticalflow_path = '../datasets/results_opticalflow_kitti/results/'
test = "../datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"

save_path = 'plots/'

if __name__ == '__main__':

    # Task 1.1
    create_folder(save_path + 'BM_optflow/')
    create_folder(save_path + 'GT_optflow/')
    create_folder(save_path + 'results/')

    sequences = read_sequences(sequences_path)
    flow_gt, flow_test = read_flow_data(gt_noc, test)

    block_size = [8]
    search_area = [4]
    step_size = [16]
    error_function=['SSD', 'MSD', 'SAD']

    for bs, sa, ss, ef in list(product(block_size, search_area, step_size, error_function)):

        print('Block size: {0}, search area: {1}, step size: {2}, error function: {3}'.format(bs,sa,ss,ef))
        opticalflow = block_matching_optical_flow(sequences[0], sequences[1], block_size=bs, search_area=sa, step_size=ss, error_function=ef, compensation='Forward')

        msen, pepn = compute_msen(flow_gt, opticalflow)

        print("MSEN: {}".format(msen))
        print("PEPN: {}".format(pepn))

        opticalflow_images = read_opticalflow(opticalflow_path)
        gt_opticalflow = read_opticalflow(gt_path)
        first_sequence = read_sequences(sequences_path)

        #plot_opticalflow_bm(opticalflow, first_sequence, title='block matching',
        #                 save_path=save_path + 'BM_optflow/')
        #plot_opticalflow_gt(gt_opticalflow, first_sequence, title='GT optical flow',
        #                 save_path=save_path + 'GT_optflow/', need_conversion=True)


    # Task 1.2

    # PyFlow
    if os.path.isfile('flow.npy'):
        pyflow = np.load('flow.npy')
    else:
        pyflow = pyflow(sequences[0], sequences[1])
    msen, pepn = compute_msen(flow_gt[1:, :], pyflow[:, 1:])
    print("MSEN: {}".format(msen))
    print("PEPN: {}".format(pepn))

    # Farneback
    farneback = farneback(sequences[0], sequences[1])
    msen, pepn = compute_msen(flow_gt, farneback)
    print("MSEN: {}".format(msen))
    print("PEPN: {}".format(pepn))

    # Lucas kanade
    lucas_kanade = lucas_kanade(sequences[0], sequences[1])
    msen, pepn = compute_msen(flow_gt, lucas_kanade)
    print("MSEN: {}".format(msen))
    print("PEPN: {}".format(pepn))


    # Task 2.1
    print("Video Stabilization with block matching")
    save_in_path = '../datasets/cat_stab/piano_in/'
    sequences = read_sequences(save_in_path)
    seq_stabilized = video_stabilization(sequences)

    # Task 2.2
    print("Point Feature Matching")
    video_path = '../datasets/cat_stab/piano.mp4'
    point_feature_matching(video_path)

    # Other implemented method in the L1Stabilizer directory written in Matlab


    # Task 3.1 in main_tracking.py
