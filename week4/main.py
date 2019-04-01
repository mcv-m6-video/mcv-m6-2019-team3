import numpy as np

from utils.optical_flow_metrics import compute_msen, read_flow_data
from utils.optical_flow_plot import plot_opticalflow, read_opticalflow, read_sequences
from block_matching import block_matching_optical_flow


test = "../datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"
gt_noc = "../datasets/data_stereo_flow/training/flow_noc/000045_10.png"

opticalflow_path = '../datasets/results_opticalflow_kitti/results/'
sequences_path = '../datasets/data_stereo_flow/training/image_0/'
gt_path = '../datasets/data_stereo_flow/training/flow_noc/'
save_path = '../plots/'


if __name__ == '__main__':

    # Task 1
    sequences = read_sequences(sequences_path)
    flow_gt, flow_test = read_flow_data(gt_noc, test)

    block_size = [4, 8, 16, 32, 64]
    search_area = [2, 4, 8, 16, 32, 64]
    step_size = [16]
    error_function=['SSD', 'MSD', 'SAD']

    for bs in block_size:
        for sa in search_area:
            for ss in step_size:
                for ef in error_function:
                    print('Block size: {0}, search area: {1}, step size: {2}, error function: {3}'.format(bs,sa,ss,ef))
                    opticalflow = block_matching_optical_flow(sequences[0], sequences[1], block_size=bs, search_area=sa, step_size=ss, error_function=ef)

                    msen, pepn = compute_msen(flow_gt, opticalflow)

                    print("MSEN: {}".format(msen))
                    print("PEPN: {}".format(pepn))
