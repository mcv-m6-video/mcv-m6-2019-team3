import os
import pickle
import cv2
from object_tracking.kalman_tracking import kalman_track_objects
from object_tracking.tracking import track_objects
from processing.background_subtraction import BackgroundSubtractor
from utils.reading import read_annotations_from_txt
from utils.filter import filtering_parked,filtering_nms
from utils.plotting import draw_video_bboxes
from optical_flow_tracking import TrackingOF
from object_tracking.centroid import CentroidTracker
import object_tracking.centroid as centroid


root_path = "../" + os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(root_path, 'datasets', 'aic19-track1-mtmc-train')
#train_path = os.path.join(dataset_path, 'train')
#train_folders = ['S01', 'S04']      # Train with S01 and S04
test_path = os.path.join(dataset_path, 'train', 'S03')
test_sequences = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

# para pruebas con una sola secuencia
test_sequences = ['c015']

# Flags
display_frames = False
export_frames = False

for sequence in test_sequences:

    with open(os.path.join("results","metrics.txt"), "a") as f:
        f.write(sequence + "\n")

    # Load video
    video_path = os.path.join(test_path, sequence, 'vdo.avi')

    # Read groundtruth
    gt_path = os.path.join(test_path, sequence, 'gt', 'gt.txt')
    groundtruth_list = read_annotations_from_txt(gt_path)
    #print(groundtruth_list)

    #################################################### Object Detection
    # Save all detection results in an array
    detections_array = []
    # State-of-the-art background subtractors
    if os.path.exists(os.path.join('pickle','detections_MOG_MOG2_GMG.pkl')):
        with open(os.path.join('pickle','detections_MOG_MOG2_GMG.pkl'), 'rb') as p:
            detectionsMOG, detectionsMOG2, detectionsGMG = pickle.load(p)
    else:
        detectionsMOG, detectionsMOG2, detectionsGMG = BackgroundSubtractor(video_path)
    # detections_array.append(detectionsMOG)
    # detections_array.append(detectionsMOG2)
    # detections_array.append(detectionsGMG)

    # Read CNN detection files
    for network in ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']:
        network_path = os.path.join(test_path, sequence, 'det', network)
        detection = read_annotations_from_txt(network_path)
        # detections_array.append(detection)

        # filtering nms
        detection = filtering_nms(detection, video_path)
        # print(detection)
        # print('STOP 2')
        # Filtering parked cars
        detection = filtering_parked(detection, video_path)
        # print(detection)
        # print('STOP 3')
        detections_array.append(detection)
        #draw_video_bboxes(video_path, groundtruth_list, detection, display=True)


    #################################################### Tracking
    for detections in detections_array:
        # Tracking by Overlap
        print("\nComputing tracking by overlap")
        with open(os.path.join("results","metrics.txt"), "a") as f:
            f.write("\nComputing tracking by overlap\n")
        detected_tracks = track_objects(video_path, detections, groundtruth_list, display=display_frames, export_frames=export_frames)

        # Kalman
        print("\nComputing Kalman tracking")
        with open(os.path.join("results","metrics.txt"), "a") as f:
            f.write("\nComputing Kalman tracking\n")
        kalman_tracks = kalman_track_objects(video_path, detections, groundtruth_list, display=display_frames, export_frames=export_frames)
        ######################## Centroid tracking
        print("\nCentroid tracking")
        with open("results/metrics.txt", "a") as f:
            f.write("\nComputing Centroid tracking\n")
        ct = CentroidTracker()
        centroid.track_objects(ct, video_path, detections, groundtruth_list, display=display_frames, export_frames=False)

        #################################################### Optical Flow
        # Optical Flow
        # print("\n Optical Flow Tracking")
        # optical_flow = TrackingOF()
        # optical_flow.track_optical_flow_sequence(video_path, False)
        # print(optical_flow.ordered_tracks[0])
        # print(optical_flow.frame_idx)
