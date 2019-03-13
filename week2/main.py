import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from processing.background_subtraction import BackgroundSubtractor, single_gaussian_model
from utils.reading import read_annotations_file
from evaluation.evaluation_funcs import compute_mAP
from utils.candidate_generation_window import plot_bboxes


video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
roi_path = '../datasets/AICity_data/train/S03/c010/roi.jpg'

# colorspace can be: None, HSV
ALPHAS = [0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
RHOS = [0.25, 0.5, 0.75, 1.]

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

    print("Best F1: {} with the alpha: {}".format(best_F1_alpha, best_alpha))
    plt.plot(F1_alpha[1], F1_alpha[0], linewidth=2.0)
    plt.xlabel('ALPHA')
    plt.ylabel('mAP')
    plt.title('Choosing alpha')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([, 160, 0, 0.03])
    plt.grid(True)
    plt.show()

    for rho in RHOS:
        detections = single_gaussian_model(roi_path, video_path, alpha=best_alpha, rho=rho, adaptive=True,
                                           export_frames=export_frames)
        print('Compute mAP0.5 for rho: {}'.format(rho))
        precision, recall, max_precision_per_step, F1, mAP = compute_mAP(groundtruth_list, detections)
        F1_rho.append([mAP, rho])
    best_rho_case = np.amax(F1_rho, axis=0)
    best_rho = best_rho_case[1]
    best_F1_rho = best_rho_case[0]
    print("Best F1: {} with the rho: {}".format(best_F1_rho, best_rho))

    print("Best F1: {} with the alpha: {}".format(best_F1_alpha, best_alpha))
    plt.plot(F1_rho[1], F1_rho[0], linewidth=2.0)
    plt.xlabel('RHO')
    plt.ylabel('mAP')
    plt.title('Choosing rho')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([, 160, 0, 0.03])
    plt.grid(True)
    plt.show()

def grid_search():
    parameters = {'alpha': ALPHAS, 'rho': RHOS}
    gs = grid_search.GridSearch()

    print('best_metric: ' + str(gs.best_score))
    print('best_params: ' + str(gs.best_params))
    scores = np.array(gs.results).reshape(len(parameters['alpha']), len(parameters['rho']))

if __name__ == "__main__":

    colorspace = None          # [None, 'HSV']
    export_frames = False
    find_best_pairs = False
    adaptive = True
    use_detections_pkl = True
    detections = []

    # Read groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_file(groundtruth_path, video_path)          # gt.txt
    print("Groundtruth loaded\n")

    # Search best alpha and rho
    if find_best_pairs:
        print("Computing best alpha and rho")
        hyperparameter_search(groundtruth_list)

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
            detections = single_gaussian_model(roi_path, video_path, alpha=2.5, rho=1, adaptive=adaptive, export_frames=export_frames)
        #print(len(detections))
        #plot_bboxes(video_path, groundtruth_list, detections)

    print('Computing mAP@0.5')
    gt_filtered = [x for x in groundtruth_list if x.frame > int(2141*0.25)]         # filter 25% of gt
    compute_mAP(gt_filtered, detections)

    # State-of-the-art background subtractors
    print("Computing detections for state-of-the-art algorithms")
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
