import os
import pickle

from evaluation.evaluation_funcs import compute_mAP, compute_mAP_track
from object_tracking.tracking import track_objects
from object_tracking.multi_camera import match_tracks
from utils.plotting import draw_video_bboxes
from utils.reading import read_annotations_file, read_homography_matrix
from object_tracking.kalman_tracking import kalman_track_objects
from utils.detection import Detection
from optical_flow_tracking import TrackingOF

# Groundtruth
video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"

# Given detections
detections_path = "../datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]
roi_path = '../datasets/AICity_data/train/S03/c010/roi.jpg'

# Own detections
mask_detections_path = "../annotations/Mask-RCNN-detections.txt"
yolo_detections_path = "../annotations/yolo_detections.txt"
yolo_fine_tuned_detections_path = "../annotations/yolo_fine_tunning_detections.txt"

if __name__ == '__main__':
    # Flags
    use_pkl = True
    display_frames = False
    export_frames = False

    # Load/Read groundtruth
    print("Getting groundtruth")
    if use_pkl and os.path.exists('groundtruth.pkl'):
        with open('groundtruth.pkl', 'rb') as p:
            print("Reading detections from groundtruth.pkl")
            groundtruth_list, tracks_gt_list = pickle.load(p)
    else:
        groundtruth_list, tracks_gt_list = read_annotations_file(groundtruth_xml_path, video_path)
        with open('groundtruth.pkl', 'wb') as f:
            pickle.dump([groundtruth_list, tracks_gt_list], f)
    # Week 3:
    # Task 1.1: Mask-RCNN Off-the-shelf
    # Read Mask-RCNN detections from file
    print("\nGetting detections")
    detections_list, _ = read_annotations_file(mask_detections_path, video_path)

    # Compute mAP
    print("\nComputing mAP")
    compute_mAP(groundtruth_list, detections_list)

    # Export bboxes
    #print("\nExporting frames")
    #draw_video_bboxes(video_path, groundtruth_list, detections_list, export_frames=export_frames)

    optical_flow = TrackingOF()
    optical_flow.track_optical_flow_sequence(video_path, False)
    # Task 2.1: Tracking by Overlap and Task 2.4: IDF1 for Multiple Object Tracking
    print("\nComputing tracking by overlap")
    detected_tracks = track_objects(video_path, detections_list, groundtruth_list, optical_flow, display=display_frames, export_frames=export_frames)

    # Compute mAP
    compute_mAP(groundtruth_list, detected_tracks)

    # Task 2.2: Kalman filter and Task 2.4: IDF1 for Multiple Object Tracking
    # print("Kalman")
    # kalman_tracks = kalman_track_objects(video_path, detections_list, groundtruth_list, display=display_frames, export_frames=export_frames)

    # # Compute mAP
    # compute_mAP(groundtruth_list, kalman_tracks)

    # Week4:
    # Optical flow tracking
    # OF_tracks = track_optical_flow_sequence(video_path,detections_list, groundtruth_list, kalman_tracks)

    # print(optical_flow.tracks)
    print(len(optical_flow.ordered_tracks))
    print(optical_flow.frame_idx)
    compute_mAP(groundtruth_list, optical_flow.tracks)
