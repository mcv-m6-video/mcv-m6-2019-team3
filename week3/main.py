import os
import pickle

from evaluation.evaluation_funcs import compute_mAP, compute_mAP_track
from object_tracking.tracking import track_objects
from object_tracking.multi_camera import match_tracks
from utils.plotting import draw_video_bboxes
from utils.reading import read_annotations_file, read_homography_matrix
from object_tracking.kalman_tracking import kalman_track_objects
from utils.detection import Detection

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
#mask_fine_tuned_detections_path = ""

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

    # Task 1.2
    # ToDo: Read fine-tuned detections, compute mAP and export frames

    # Task 2.1: Tracking by Overlap and Task 2.4: IDF1 for Multiple Object Tracking
    print("\nComputing tracking by overlap")
    detected_tracks = track_objects(video_path, detections_list, groundtruth_list, display=display_frames, export_frames=export_frames)

    # Compute mAP
    compute_mAP(groundtruth_list, detected_tracks)

    # Task 2.2: Kalman filter and Task 2.4: IDF1 for Multiple Object Tracking
    print("Kalman")
    kalman_tracks = kalman_track_objects(video_path, detections_list, groundtruth_list, display=display_frames, export_frames=export_frames)

    # Compute mAP
    compute_mAP(groundtruth_list, kalman_tracks)




    # Task 3: CVPR challenge
    cameras_path = ["../datasets/AICity_data/train/S01/c001/", "../datasets/AICity_data/train/S01/c002/"]
    video_challenge_path = "vdo.avi"
    detections_challenge_path = "det/"
    detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]
    groundtruth_challenge_path = "gt/gt.txt"
    homography_path = "calibration.txt"
    timestamps = [0, 1.640]
    framenum = [1955, 2110]
    fps = 10

    detected_tracks = {}
    homography_cameras = {}

    for cam_num, camera in enumerate(cameras_path):
        #for detector in detectors:
        #print(detector)
        #detections_list, _ = read_annotations_file(camera + detections_challenge_path + detector, camera + video_challenge_path)
        groundtruth_list, _ = read_annotations_file(camera + groundtruth_challenge_path, camera + video_challenge_path)
        homography_cameras[cam_num] = read_homography_matrix(camera + homography_path)

        print("\nComputing tracking by overlap")
        detected_tracks[cam_num] = track_objects(camera + video_challenge_path, groundtruth_list, groundtruth_list,
                                        display=display_frames, export_frames=export_frames, idf1=False)

        # Compute mAP
        compute_mAP(groundtruth_list, detected_tracks[cam_num])

    match_tracks(detected_tracks, homography_cameras, timestamps, framenum, fps, cameras_path[0] + video_challenge_path, cameras_path[1] + video_challenge_path)



