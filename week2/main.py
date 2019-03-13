import os
import pickle
import numpy as np

from evaluation.evaluation_funcs import compute_mAP
from processing.background_subtraction import BackgroundSubtractor, single_gaussian_model, hyperparameter_search, gridsearch
from utils.reading import read_annotations_file

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
roi_path = '../datasets/AICity_data/train/S03/c010/roi.jpg'

if __name__ == "__main__":

    colorspace = None               # [None, 'HSV']
    adaptive = True
    use_detections_pkl = True
    export_frames = False

    # Find best alpha and rho
    find_best_pairs = False
    grid_search = False
    no_hyperparameter = False
    detections = []

    # Read groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_file(groundtruth_path, video_path)          # gt.txt
    gt_filtered = [x for x in groundtruth_list if x.frame > int(2141 * 0.25)]       # filter 25% of gt
    print("Groundtruth loaded\n")

    # Search best alpha and rho
    if find_best_pairs:
        print("Computing best alpha and rho")
        hyperparameter_search(gt_filtered, video_path, roi_path)

    if grid_search:
        print("Computing best alpha and rho with gridsearch")
        gridsearch(gt_filtered, video_path, roi_path)

    if no_hyperparameter:
        # Check colorspace
        if colorspace is not None:
            if colorspace.lower() == "hsv":
                print("HSV")
                # Load detections
                if use_detections_pkl and os.path.exists('detections_h.pkl'):
                    with open('detections_h.pkl', 'rb') as p:
                        print("Reading detections from detections_h.pkl")
                        detections = pickle.load(p)
                        print("Detections loaded\n")
                else:
                    # Compute detections
                    # This function lasts about 10 minutes
                    detections = single_gaussian_model(roi_path, video_path, alpha=1.25, rho=1, adaptive=adaptive, export_frames=export_frames, only_h=True)
        else:
            # Load detections
            if use_detections_pkl and os.path.exists('detections.pkl'):
                with open('detections.pkl', 'rb') as p:
                    print("Reading detections from detections.pkl")
                    detections = pickle.load(p)
                    print("Detections loaded\n")
            else:
                # Compute detections
                # This function lasts about 10 minutes
                detections = single_gaussian_model(roi_path, video_path, alpha=2.5, rho=1, adaptive=adaptive, export_frames=export_frames)

    else:
        # Load detections
        if use_detections_pkl and os.path.exists('detections.pkl'):
            with open('detections.pkl', 'rb') as p:
                print("Reading detections from detections.pkl")
                detections = pickle.load(p)
                print("Detections loaded\n")
        else:
            # Compute detections
            # This function lasts about 10 minutes
            detections = single_gaussian_model(roi_path, video_path, alpha=2.5, rho=1, adaptive=adaptive, export_frames=export_frames)
            #plot_bboxes(video_path, groundtruth_list, detections)

    print('Compute mAP@0.5')
    compute_mAP(gt_filtered, detections)

    # State-of-the-art background subtractors
    detectionsMOG, detectionsMOG2, detectionsGMG = BackgroundSubtractor(video_path, export_frames=export_frames)
    print('mAP0.5 for MOG:')
    detectionsMOG_filtered = [x for x in detectionsMOG if x.frame > int(2141 * 0.25)]
    compute_mAP(gt_filtered, detectionsMOG_filtered)
    print('mAP0.5 for MOG2:')
    detectionsMOG2_filtered = [x for x in detectionsMOG2 if x.frame > int(2141 * 0.25)]
    compute_mAP(gt_filtered, detectionsMOG2_filtered)
    print('mAP0.5 for GMG:')
    detectionsGMG_filtered = [x for x in detectionsGMG if x.frame > int(2141 * 0.25)]
    compute_mAP(gt_filtered, detectionsGMG_filtered)
