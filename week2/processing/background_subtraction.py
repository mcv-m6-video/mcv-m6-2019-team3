import cv2
import os
import pickle

import numpy as np
from tqdm import tqdm

from utils.morphology_utils import morphological_filtering
from utils.candidate_generation_window import visualize_boxes, candidate_generation_window_ccl

#from week1.utils.reading import read_annotations_file

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

        # Get groundtruth and detections from frame n
        #gt_on_frame = [x for x in groundtruth_list if x.frame == n]
        #gt_bboxes = [o.bbox for o in gt_on_frame]

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


def get_fg_mask_single_gaussian_model(video_path, first_frame, model_mean, model_std, alpha, rho, adaptive=False, only_h=False):
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
            foreground[n_frame-first_frame-1, :, :] = morphological_filtering(get_frame_mask_single_gaussian_model
                                                                              (image, model_mean, model_std, alpha))
            window_candidates = candidate_generation_window_ccl(n_frame, foreground[n_frame-first_frame-1, :, :])
            detections.extend(window_candidates)
            #visualize_boxes(image, window_candidates)

            if adaptive:
                model_mean = rho*image + (1-rho)*model_mean
                model_std = np.sqrt(rho*(image - model_mean)**2 + (1-rho)*model_std**2)

        pbar.update(1)
        n_frame +=1

    pbar.close()

    return foreground, detections


def single_gaussian_model(video_path, alpha, rho, adaptive=False, export_frames=False, only_h=False):

    print('Computing Gaussian model...')
    if only_h:
        filename_mstd = 'mean_std_h.pkl'
    else:
        filename_mstd = 'mean_std.pkl'
    if os.path.exists(filename_mstd):
        with open(filename_mstd, 'rb') as p:
            mean, std = pickle.load(p)
    else:
        mean, std = get_pixels_single_gaussian_model(video_path)
        with open(filename_mstd, 'wb') as f:
            pickle.dump([mean, std], f)

    print('Gaussian computed for pixels')
    print('\nExtracting Background...')
    bg, detections = get_fg_mask_single_gaussian_model(video_path, first_frame=int(2141 * 0.25), model_mean=mean, model_std=std,
                                            alpha=alpha, rho=rho, adaptive=adaptive, only_h=only_h)
    print('Extracted background with shape {}'.format(bg.shape))

    if export_frames:
        i = int(2141 * 0.25)
        for frame in bg:
            new_image = frame.astype(np.uint8)
            new_image = cv2.resize(new_image, (0, 0), fx=0.3, fy=0.3)
            cv2.imwrite('output_frames/single_gaussian/frame_{:04d}.png'.format(i), new_image.astype('uint8') * 255)
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

    fgbg_MOG2 = cv2.createBackgroundSubtractorMOG2()
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

    while capture.isOpened():
        valid, frame = capture.read()
        if not valid:
            break
        images.append(frame)

        # Algorithms
        maskMOG = BackgroundSubtractorMOG(fgbg_MOG, frame, i, export_frames)
        maskMOG2 = BackgroundSubtractorMOG2(fgbg_MOG2, frame, i, export_frames)
        maskGMG = BackgroundSubtractorGMG(fgbg_GMG, frame, i, export_frames)

        maskMOG = morphological_filtering(maskMOG)
        window_candidatesMOG = candidate_generation_window_ccl(i, maskMOG)
        detectionsMOG.extend(window_candidatesMOG)

        maskMOG2 = morphological_filtering(maskMOG2)
        window_candidatesMOG2 = candidate_generation_window_ccl(i, maskMOG2)
        detectionsMOG.extend(window_candidatesMOG2)

        maskGMG = morphological_filtering(maskGMG)
        window_candidatesGMG = candidate_generation_window_ccl(i, maskGMG)
        detectionsMOG.extend(window_candidatesGMG)

        i += 1

    capture.release()
    cv2.destroyAllWindows()

    return detectionsMOG, detectionsMOG2, detectionsGMG


def BackgroundSubtractorMOG(fgbg_MOG, frame, i, export_frames=False):
    fgmask = fgbg_MOG.apply(frame)
    fgmask = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
    if export_frames:
        cv2.imwrite('output_frames/MOG/frame_{:04d}.jpg'.format(i), fgmask)
    return fgmask

def BackgroundSubtractorMOG2(fgbg_MOG2, frame, i, export_frames=False):
    fgmask = fgbg_MOG2.apply(frame)
    fgmask = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
    if export_frames:
        cv2.imwrite('output_frames/MOG2/frame_{:04d}.jpg'.format(i), fgmask)
    return fgmask

def BackgroundSubtractorGMG(fgbg_GMG, frame, i, export_frames=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    fgmask = fgbg_GMG.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
    if export_frames:
        cv2.imwrite('output_frames/GMG/frame_{:04d}.jpg'.format(i), fgmask)
    return fgmask
