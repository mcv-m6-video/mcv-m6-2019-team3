from week2.processing.background_subtraction import BackgroundSubtractor, single_gaussian_model
from week2.utils.reading import read_annotations_file
from week2.evaluation.evaluation_funcs import compute_mAP

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"

if __name__ == "__main__":

    # Gaussian modelling
    detections = single_gaussian_model(video_path, alpha=2.5, rho=1, adaptive=True)

    #Evaluate against groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_file(groundtruth_path)          # gt.txt
    print('Compute mAP0.5')
    compute_mAP(groundtruth_list, detections)


    # State-of-the-art background subtractors
    #BackgroundSubtractor(video_path, export_frames=True)
