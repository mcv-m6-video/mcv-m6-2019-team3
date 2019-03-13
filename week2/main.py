import os
import pickle
import numpy as np

from processing.background_subtraction import BackgroundSubtractor, single_gaussian_model
from utils.reading import read_annotations_file
from evaluation.evaluation_funcs import compute_mAP
from utils.plotting import plot2D,plot3D



video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
roi_path = '../datasets/AICity_data/train/S03/c010/roi.jpg'

# colorspace can be: None, HSV
ALPHAS = [0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
RHOS = [0.25, 0.5, 0.75, 1.]

ALPHAS = [2.3, 2.5, 2.7]
RHOS = [0.8, 0.9, 1.0]
#mAP = [0.316, 0.516, 0.531, 0.251, 0.529, 0.531, 0.186, 0.524, 0.563]

def hyperparameter_search(groundtruth_list):

    F1_alpha = []
    F1_rho = []

    for alpha in ALPHAS:
        detections = single_gaussian_model(roi_path, video_path, alpha=alpha, rho=1, adaptive=False, export_frames=export_frames)
        print('Compute mAP0.5 for alpha: {}'.format(alpha))
        precision, recall, max_precision_per_step, F1, mAP = compute_mAP(groundtruth_list, detections)
        F1_alpha.append([mAP, alpha])
    best_alpha_case = np.amax(F1_alpha, axis=0)
    best_alpha = best_alpha_case[1]
    best_F1_alpha = best_alpha_case[0]

    print("Best mAP: {} with the alpha: {}".format(best_F1_alpha, best_alpha))

    X = np.array(F1_alpha)[:,1]

    Y = np.array(F1_alpha)[:,0]

    plot2D(X, Y, "ALPHA", "mAP", "Choosing alpha")

    for rho in RHOS:
        detections = single_gaussian_model(roi_path, video_path, alpha=best_alpha, rho=rho, adaptive=True,
                                           export_frames=export_frames)
        print('Compute mAP0.5 for rho: {}'.format(rho))
        precision, recall, max_precision_per_step, F1, mAP = compute_mAP(groundtruth_list, detections)
        F1_rho.append([mAP, rho])
    best_rho_case = np.amax(F1_rho, axis=0)
    best_rho = best_rho_case[1]
    best_F1_rho = best_rho_case[0]

    print("Best mAP: {} with the rho: {}".format(best_F1_rho, best_rho))

    X = np.array(F1_rho)[:,1]

    Y = np.array(F1_rho)[:,0]

    plot2D(X, Y, "RO", "mAP", "Choosing ro")

def gridsearch(groundtruth_list):
    mAP_total = []
    Z = np.zeros([len(ALPHAS),len(RHOS)])
    for i,alpha in enumerate(ALPHAS):
        for j,rho in enumerate(RHOS):
            detections = single_gaussian_model(roi_path, video_path, alpha=alpha, rho=rho, adaptive=True,
                                               export_frames=False)
            print('Compute mAP0.5 for alpha {} and rho {}'.format(alpha, rho))
            precision, recall, max_precision_per_step, F1, mAP = compute_mAP(groundtruth_list, detections)
            mAP_total.append([mAP, alpha, rho])
            Z[i,j] = mAP

    best_case = np.amax(mAP_total, axis=0)

    print("Best mAP: {} with alpha: {} and rho {}".format(best_case[0], best_case[1], best_case[2]))

    # Plot grid search
    X, Y = np.meshgrid(RHOS, ALPHAS)

    plot3D(X,Y,Z)



if __name__ == "__main__":

    colorspace = None               # [None, 'HSV']
    adaptive = False
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
        hyperparameter_search(gt_filtered)

    if grid_search:
        print("Computing best alpha and rho with gridsearch")
        gridsearch(gt_filtered)

    if no_hyperparameter:
        # Check colorspace
        if colorspace is not None:
            if colorspace.lower() == "hsv":
                print("hsv")
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
