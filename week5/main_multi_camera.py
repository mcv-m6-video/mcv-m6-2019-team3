from itertools import product
import numpy as np
import os
import pickle

# Tracking
from evaluation.evaluation_funcs import compute_mAP
from object_tracking.tracking import track_objects
from object_tracking.multi_camera import match_tracks_by_frame, create_dataset, predict_bbox, bboxes_correspondences, match_tracks
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

    cameras_path = ["../datasets/aic19-track1-mtmc-train/train/S01/c001/", "../datasets/aic19-track1-mtmc-train/train/S01/c002/"]
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

    tracked_detections = {}
    tracks_by_camera = {}
    homography_cameras = {}
    groundtruth_list = {}


    for cam_num, camera in enumerate(cameras_path):
        #detections_list, _ = read_annotations_file(camera + detections_challenge_path + detector, camera + video_challenge_path)
        groundtruth_list[cam_num], _ = read_annotations_file(camera + groundtruth_challenge_path, camera + video_challenge_path)
        homography_cameras[cam_num] = read_homography_matrix(homography_path_start + camera[(len(camera)-5):] + homography_path)

        print("\nComputing tracking by overlap")
        if load_pkl and os.path.exists('detections'+str(cam_num)+'.pkl') and os.path.exists('tracks'+str(cam_num)+'.pkl'):
            with open('detections' + str(cam_num)+'.pkl', 'rb') as p:
                print("Reading tracked detections from pkl")
                tracked_detections[cam_num] = pickle.load(p)
                print("Tracked detections loaded\n")

            with open('tracks' + str(cam_num)+'.pkl', 'rb') as p:
                print("Reading tracks from pkl")
                tracks_by_camera[cam_num] = pickle.load(p)
                print("Tracks loaded\n")

        else:

            tracked_detections[cam_num], tracks_by_camera[cam_num] = track_objects(camera + video_challenge_path, groundtruth_list[cam_num], groundtruth_list[cam_num],
                                            display=display_frames, export_frames=export_frames, idf1=False, name_pkl=str(cam_num))

        # Compute mAP
        compute_mAP(groundtruth_list[cam_num], tracked_detections[cam_num])

    #correspondences = bboxes_correspondences(groundtruth_list, timestamps, framenum, fps)
    match_tracks(tracked_detections, tracks_by_camera, homography_cameras, timestamps, framenum, fps, cameras_path[0] + video_challenge_path, cameras_path[1] + video_challenge_path)
    #match_tracks_by_frame(tracked_detections, homography_cameras, timestamps, framenum, fps, cameras_path[0] + video_challenge_path, cameras_path[1] + video_challenge_path, correspondences)

