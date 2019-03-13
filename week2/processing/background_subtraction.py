import cv2
import os
import pickle

import numpy as np
from tqdm import tqdm

from evaluation.evaluation_funcs import compute_mAP
from utils.candidate_generation_window import candidate_generation_window_ccl, visualize_boxes
from utils.morphology_utils import morphological_filtering
from utils.plotting import plot2D,plot3D

############################################
########### Single gaussian method
############################################

def get_pixels_single_gaussian_model(video_path, last_frame=int(2141*0.25), only_h=False):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    pbar = tqdm(total=last_frame+1)

    while capture.isOpened() and n_frame <= last_frame:
        valid, image = capture.read()
        if not valid:
            break

        if n_frame==0:
            gaussians = np.zeros((last_frame+1, image.shape[0], image.shape[1]))
        if only_h:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = image[:,:,0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gaussians[n_frame, :, :] = image

        pbar.update(1)
        n_frame += 1

    pbar.close()

    print("\nComputing mean...")
    gauss_mean = gaussians.mean(axis=0)
    print("Computing standard deviation...")
    gauss_std = gaussians.std(axis=0)

    return gauss_mean, gauss_std


def get_frame_mask_single_gaussian_model(img, model_mean, model_std, alpha):
    return abs(img - model_mean) >= alpha*(model_std+2)     # Foreground


def get_fg_mask_single_gaussian_model(roi, video_path, first_frame, model_mean, model_std, alpha, rho, adaptive=False, only_h=False, min_h=100, max_h=480, min_w=100, max_w=600, min_ratio=0.4, max_ratio=1.30):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0
    detections = []
    pbar = tqdm(total=2141)

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break

        if n_frame == first_frame:
            foreground = np.zeros((2141 - first_frame, image.shape[0], image.shape[1]))
        if n_frame > first_frame:
            if only_h:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image = image[:,:,0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            foreground[n_frame-first_frame-1, :, :] = morphological_filtering(roi*get_frame_mask_single_gaussian_model
                                                                              (image, model_mean, model_std, alpha))

            window_candidates = candidate_generation_window_ccl(n_frame, foreground[n_frame-first_frame-1, :, :], min_h, max_h, min_w, max_w, min_ratio, max_ratio)
            detections.extend(window_candidates)
            #visualize_boxes(image, window_candidates, window_candidates)

            if adaptive:
                model_mean = rho*image + (1-rho)*model_mean
                model_std = np.sqrt(rho*(image - model_mean)**2 + (1-rho)*model_std**2)

        pbar.update(1)
        n_frame +=1

    pbar.close()

    return foreground, detections


def single_gaussian_model(roi_path, video_path, alpha, rho, adaptive=False, export_frames=False, only_h=False, use_detections_pkl=True):
    roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY)

    print('Computing Gaussian model...')

    if only_h:
        filename_mstd = 'mean_std_h.pkl'
    else:
        filename_mstd = 'mean_std.pkl'

    # Load mean and std
    if use_detections_pkl and os.path.exists(filename_mstd):
        with open(filename_mstd, 'rb') as p:
            mean, std = pickle.load(p)
    else:
        mean, std = get_pixels_single_gaussian_model(video_path)
        with open(filename_mstd, 'wb') as f:
            pickle.dump([mean, std], f)

    print('Gaussian computed for pixels\n')

    print('Extracting Background...')
    bg, detections = get_fg_mask_single_gaussian_model(roi, video_path, first_frame=int(2141 * 0.25), model_mean=mean, model_std=std,
                                            alpha=alpha, rho=rho, adaptive=adaptive, only_h=only_h)
    print('Extracted background with shape {}'.format(bg.shape))

    if export_frames:
        i = int(2141 * 0.25)
        for frame in bg:
            new_image = frame.astype(np.uint8)
            new_image = cv2.resize(new_image, (0, 0), fx=0.3, fy=0.3)
            cv2.imwrite('output_frames/single_gaussian_constraint/frame_{:04d}.png'.format(i), new_image.astype('uint8') * 255)
            i += 1

    if only_h:
        with open('detections_h.pkl', 'wb') as f:
            pickle.dump(detections, f)
    else:
        with open('detections.pkl', 'wb') as f:
            pickle.dump(detections, f)

    return detections

############################################
########### State-of-the-art methods
############################################

def BackgroundSubtractor(video_path, export_frames=False):
    capture = cv2.VideoCapture(video_path)

    fgbg_MOG = cv2.bgsegm.createBackgroundSubtractorMOG()
    if not os.path.exists('output_frames/MOG/'):
        os.mkdir('output_frames/MOG/')

    fgbg_MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    if not os.path.exists('output_frames/MOG2/'):
        os.mkdir('output_frames/MOG2/')

    fgbg_GMG = cv2.bgsegm.createBackgroundSubtractorGMG()
    if not os.path.exists('output_frames/GMG/'):
        os.mkdir('output_frames/GMG/')

    i = 0
    images = []
    detectionsMOG = []
    detectionsMOG2 = []
    detectionsGMG = []

    pbar = tqdm(total=2140)

    while capture.isOpened():
        valid, frame = capture.read()
        if not valid:
            break
        #images.append(frame)

        ### Algorithms
        # MOG
        maskMOG = BackgroundSubtractorMOG(fgbg_MOG, frame, i, export_frames)
        maskMOG = morphological_filtering(maskMOG)
        window_candidatesMOG = candidate_generation_window_ccl(i, maskMOG)
        detectionsMOG.extend(window_candidatesMOG)

        # MOG2
        maskMOG2 = BackgroundSubtractorMOG2(fgbg_MOG2, frame, i, export_frames)
        maskMOG2 = morphological_filtering(maskMOG2)
        window_candidatesMOG2 = candidate_generation_window_ccl(i, maskMOG2)
        detectionsMOG2.extend(window_candidatesMOG2)

        # GMG
        maskGMG = BackgroundSubtractorGMG(fgbg_GMG, frame, i, export_frames)
        maskGMG = morphological_filtering(maskGMG)
        window_candidatesGMG = candidate_generation_window_ccl(i, maskGMG)
        detectionsGMG.extend(window_candidatesGMG)

        pbar.update(1)
        i += 1

    pbar.close()
    capture.release()
    cv2.destroyAllWindows()

    with open('detections_MOG_MOG2_GMG.pkl', 'wb') as f:
        pickle.dump([detectionsMOG, detectionsMOG2, detectionsGMG], f)

    return detectionsMOG, detectionsMOG2, detectionsGMG


def BackgroundSubtractorMOG(fgbg_MOG, frame, i, export_frames=False):
    fgmask = fgbg_MOG.apply(frame)
    if export_frames:
        fgmask_out = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
        cv2.imwrite('output_frames/MOG/frame_{:04d}.png'.format(i), fgmask_out.astype('uint8') * 255)
    return fgmask

def BackgroundSubtractorMOG2(fgbg_MOG2, frame, i, export_frames=False):
    fgmask = fgbg_MOG2.apply(frame)
    if export_frames:
        fgmask_out = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
        cv2.imwrite('output_frames/MOG2/frame_{:04d}.png'.format(i), fgmask_out.astype('uint8') * 255)
    return fgmask

def BackgroundSubtractorGMG(fgbg_GMG, frame, i, export_frames=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    fgmask = fgbg_GMG.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    if export_frames:
        fgmask_out = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
        cv2.imwrite('output_frames/GMG/frame_{:04d}.png'.format(i), fgmask_out.astype('uint8') * 255)
    return fgmask

############################################
########### Alpha & Rho Optimization
############################################

def hyperparameter_search(groundtruth_list, video_path, roi_path, ):

    ALPHAS = [0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
    RHOS = [0.25, 0.5, 0.75, 1.]
    F1_alpha = []
    F1_rho = []

    for alpha in ALPHAS:
        detections = single_gaussian_model(roi_path, video_path, alpha=alpha, rho=1, adaptive=False)
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
        detections = single_gaussian_model(roi_path, video_path, alpha=best_alpha, rho=rho, adaptive=True)
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

def gridsearch(groundtruth_list, video_path, roi_path):

    ALPHAS = [2.3, 2.5, 2.7]
    RHOS = [0.8, 0.9, 1.0]

    mAP_total = []
    Z = np.zeros([len(ALPHAS),len(RHOS)])
    for i,alpha in enumerate(ALPHAS):
        for j,rho in enumerate(RHOS):
            detections = single_gaussian_model(roi_path, video_path, alpha=alpha, rho=rho, adaptive=True)
            print('Compute mAP0.5 for alpha {} and rho {}'.format(alpha, rho))
            precision, recall, max_precision_per_step, F1, mAP = compute_mAP(groundtruth_list, detections)
            mAP_total.append([mAP, alpha, rho])
            Z[i,j] = mAP

    best_case = np.amax(mAP_total, axis=0)

    print("Best mAP: {} with alpha: {} and rho {}".format(best_case[0], best_case[1], best_case[2]))

    # Plot grid search
    X, Y = np.meshgrid(RHOS, ALPHAS)

    plot3D(X,Y,Z)