from itertools import product
import numpy as np
import os
import pickle

# Optical flow
from block_matching import block_matching_optical_flow
from optical_flow import pyflow, farneback, lucas_kanade
from utils.optical_flow_metrics import compute_msen, read_flow_data
from utils.optical_flow_plot import plot_opticalflow_gt, plot_opticalflow_bm, read_opticalflow, read_sequences
from utils.plotting import create_folder
from video_stabilizer import video_stabilization, point_feature_matching

# Tracking
from evaluation.evaluation_funcs import compute_mAP
from object_tracking.tracking import track_objects
from object_tracking.multi_camera import match_tracks, create_dataset, predict_bbox, bboxes_correspondences
from utils.plotting import draw_video_bboxes
from utils.reading import read_annotations_file, read_homography_matrix
from object_tracking.kalman_tracking import kalman_track_objects
from utils.detection import Detection

sequences_path = '../datasets/data_stereo_flow/training/image_0/'

gt_path = '../datasets/data_stereo_flow/training/flow_noc/'
gt_noc = "../datasets/data_stereo_flow/training/flow_noc/000045_10.png"

opticalflow_path = '../datasets/results_opticalflow_kitti/results/'
test = "../datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"

save_path = 'plots/'


if __name__ == '__main__':

    # # Task 1.1
    # create_folder(save_path + 'BM_optflow/')
    # create_folder(save_path + 'GT_optflow/')
    # create_folder(save_path + 'results/')
    #
    # sequences = read_sequences(sequences_path)
    # flow_gt, flow_test = read_flow_data(gt_noc, test)
    #
    # block_size = [2, 4, 8, 16, 32, 64]
    # search_area = [4]
    # step_size = [16]
    # error_function=['SSD', 'MSD', 'SAD']
    #
    # for bs, sa, ss, ef in list(product(block_size, search_area, step_size, error_function)):
    #
    #     print('Block size: {0}, search area: {1}, step size: {2}, error function: {3}'.format(bs,sa,ss,ef))
    #     opticalflow = block_matching_optical_flow(sequences[0], sequences[1], block_size=bs, search_area=sa, step_size=ss, error_function=ef, compensation='Forward')
    #
    #     msen, pepn = compute_msen(flow_gt, opticalflow)
    #
    #     print("MSEN: {}".format(msen))
    #     print("PEPN: {}".format(pepn))
    #
    #     opticalflow_images = read_opticalflow(opticalflow_path)
    #     gt_opticalflow = read_opticalflow(gt_path)
    #     first_sequence = read_sequences(sequences_path)
    #
    #     #plot_opticalflow_bm(opticalflow, first_sequence[0], title='block matching',
    #     #                 save_path=save_path + 'BM_optflow/')
    #     #plot_opticalflow_gt(gt_opticalflow, first_sequence, title='GT optical flow',
    #     #                 save_path=save_path + 'GT_optflow/', need_conversion=True)
    #
    #
    # # Task 1.2
    #
    # # PyFlow
    # if os.path.isfile('flow.npy'):
    #     pyflow = np.load('flow.npy')
    # else:
    #     pyflow = pyflow(sequences[0], sequences[1])
    # msen, pepn = compute_msen(flow_gt[1:, :], pyflow[:, 1:])
    # print("MSEN: {}".format(msen))
    # print("PEPN: {}".format(pepn))
    #
    # # Farneback
    # farneback = farneback(sequences[0], sequences[1])
    # msen, pepn = compute_msen(flow_gt, farneback)
    # print("MSEN: {}".format(msen))
    # print("PEPN: {}".format(pepn))
    #
    # # Lucas kanade
    # lucas_kanade = lucas_kanade(sequences[0], sequences[1])
    # msen, pepn = compute_msen(flow_gt, lucas_kanade)
    # print("MSEN: {}".format(msen))
    # print("PEPN: {}".format(pepn))
    #
    #
    # # Task 2.1
    # print("Video Stabilization with block matching")
    # save_in_path = '../datasets/cat_stab/piano_in/'
    # sequences = read_sequences(save_in_path)
    # seq_stabilized = video_stabilization(sequences)
    #
    # # Task 2.2
    # print("Point Feature Matching")
    # video_path = '../datasets/cat_stab/piano.mp4'
    # point_feature_matching(video_path)
    #
    # # Other implemented method in the L1Stabilizer directory written in Matlab


    # Task 3.1 in main_tracking.py

    # Task 3.2: CVPR challenge
    cameras_path = ["../datasets/AICity_data/train/S01/c001/", "../datasets/AICity_data/train/S01/c002/"]
    video_challenge_path = "vdo.avi"
    detections_challenge_path = "det/"
    detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]
    groundtruth_challenge_path = "gt/gt.txt"
    homography_path_start = "../datasets/calibration/"
    homography_path = "calibration.txt"
    timestamps = [0, 1.640]
    framenum = [1955, 2110]
    fps = 10
    display_frames = False
    export_frames = False
    load_pkl = True

    detected_tracks = {}
    homography_cameras = {}
    groundtruth_list = {}


    for cam_num, camera in enumerate(cameras_path):
        #detections_list, _ = read_annotations_file(camera + detections_challenge_path + detector, camera + video_challenge_path)
        groundtruth_list[cam_num], _ = read_annotations_file(camera + groundtruth_challenge_path, camera + video_challenge_path)
        homography_cameras[cam_num] = read_homography_matrix(homography_path_start + camera[(len(camera)-5):] + homography_path)

        print("\nComputing tracking by overlap")
        if load_pkl and os.path.exists(str(cam_num)+'.pkl'):
            with open(str(cam_num)+'.pkl', 'rb') as p:
                print("Reading tracks from pkl")
                detected_tracks[cam_num] = pickle.load(p)
                print("Tracks loaded\n")
        else:

            detected_tracks[cam_num] = track_objects(camera + video_challenge_path, groundtruth_list[cam_num], groundtruth_list[cam_num],
                                            display=display_frames, export_frames=export_frames, idf1=False, name_pkl=str(cam_num))

        # Compute mAP
        compute_mAP(groundtruth_list[cam_num], detected_tracks[cam_num])

    correspondences = bboxes_correspondences(groundtruth_list, timestamps, framenum, fps)
    match_tracks(detected_tracks, homography_cameras, timestamps, framenum, fps, cameras_path[0] + video_challenge_path, cameras_path[1] + video_challenge_path, correspondences)

