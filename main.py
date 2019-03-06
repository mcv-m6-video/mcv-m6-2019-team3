from evaluation.evaluation_funcs import compute_IoU, compute_mAP, plot_precision_recall_curve
from utils.reading import read_annotations_file
from utils.modify_detections import obtain_modified_detections
from evaluation.temporal_analysis import plotIoU, plotF1, plotIoU_by_frame, plotF1_by_frame


video_path = "./datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_path = "./datasets/AICity_data/train/S03/c010/gt/gt.txt"
detections_path = "./datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]

annotated_groundtruth = "./annotations/Anotation_740-1090_AICITY_S03_C010.xml"

if __name__ == "__main__":

    # Get groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_file(groundtruth_path)          # gt.txt
    #groundtruth_list = read_annotations_file(annotated_groundtruth)    # Our .xml

    # Get detections
    print("Getting detections")
    detections_list = read_annotations_file(groundtruth_path)

    # Get modified detections
    print("Computing modified detections")
    detections_modified = obtain_modified_detections(detections_list)

    # Compute IoU
    print("\nComputing IoU")
    IoUFrames, F1Frames= compute_IoU(video_path, groundtruth_list, detections_list)
    plotIoU(IoUFrames, "./plots/IOUplots")
    plotIoU_by_frame(IoUFrames, "./plots/IOUplots")
    plotF1(F1Frames, "./plots/F1plots")
    plotF1_by_frame(IoUFrames, "./plots/F1plots")

    # Repeat with modified detections
    print("Computing IoU with modified detections")
    IoUFrames, F1Frames = compute_IoU(video_path, groundtruth_list, detections_modified)
    plotIoU(IoUFrames, "./plots/IOUplots_noise")
    plotIoU_by_frame(IoUFrames, "./plots/IOUplots_noise")
    plotF1(F1Frames, "./plots/F1plots_noise")
    plotF1_by_frame(IoUFrames, "./plots/F1plots_noise")

    # T1.2 Compute mAP
    print("\nComputing mAP")
    precision, recall = compute_mAP(groundtruth_list, detections_list)
    plot_precision_recall_curve(precision, recall, 'gt')

    # Repeat with modified detections
    print("Computing mAP with modified detections")
    precision, recall = compute_mAP(groundtruth_list, detections_modified)
    plot_precision_recall_curve(precision, recall, 'modified_gt')

    # T1.3 Calculate mAP with different detectors
    for detector in detectors:
        print(detector)
        detections_list = read_annotations_file(detections_path + detector)
        precision, recall = compute_mAP(groundtruth_list, detections_list)
        plot_precision_recall_curve(precision, recall, detector)
