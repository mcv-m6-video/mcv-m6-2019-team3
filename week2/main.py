import os
import pickle

from processing.background_subtraction import BackgroundSubtractor, single_gaussian_model
from utils.reading import read_annotations_file
from evaluation.evaluation_funcs import compute_mAP
from utils.candidate_generation_window import plot_bboxes

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"

if __name__ == "__main__":
    export_frames = False

    # Gaussian modelling
    if os.path.exists('detections.pkl'):
        with open('detections.pkl', 'rb') as p:
            detections = pickle.load(p)
    else:
        # This function lasts about 10 minutes
        detections = single_gaussian_model(video_path, alpha=2.5, rho=1, adaptive=True, export_frames=export_frames)

    #Evaluate against groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_file(groundtruth_xml_path, video_path)          # gt.txt
    print(groundtruth_list)

    plot_bboxes(video_path, groundtruth_list, detections)

    print('Compute mAP0.5')
    compute_mAP(groundtruth_list, detections)


    # State-of-the-art background subtractors
    #BackgroundSubtractor(video_path, export_frames=export_frames)
