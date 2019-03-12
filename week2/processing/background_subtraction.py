import cv2
import os

import numpy as np

from week2.utils.morphology_utils import morphological_filtering
from week2.utils.candidate_generation_window import visualize_boxes, candidate_generation_window_ccl

#from week1.utils.reading import read_annotations_file

def get_pixels_single_gaussian_model(video_path, last_frame=int(2141*0.25)):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened() and n_frame <= last_frame:
        valid, image = capture.read()
        if not valid:
            break
        if n_frame==0:
            gaussians = np.zeros((image.shape[0], image.shape[1], last_frame+1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gaussians[:, :, n_frame] = image

        # Get groundtruth and detections from frame n
        #gt_on_frame = [x for x in groundtruth_list if x.frame == n]
        #gt_bboxes = [o.bbox for o in gt_on_frame]

        n_frame += 1

    gauss_mean = gaussians.mean(axis=2)
    gauss_std = gaussians.std(axis=2)

    return gauss_mean, gauss_std


def get_frame_mask_single_gaussian_model(img, model_mean, model_std, alpha):
    foreground = (abs(img - model_mean) >= alpha*(model_std+2))

    return foreground


def get_fg_mask_single_gaussian_model(video_path, first_frame, model_mean, model_std, alpha, rho, adaptive=False):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        if n_frame == first_frame:
            foreground = np.zeros((image.shape[0], image.shape[1], 2141 - first_frame))
        if n_frame > first_frame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            foreground[:, :, n_frame-first_frame-1] = morphological_filtering(get_frame_mask_single_gaussian_model
                                                                              (image, model_mean, model_std, alpha))
            window_candidates = candidate_generation_window_ccl(foreground[:, :, n_frame-first_frame-1])
            visualize_boxes(image, window_candidates)

            if adaptive:
                model_mean = rho*image + (1-rho)*model_mean
                model_std = np.sqrt(rho*(image - model_mean)**2 + (1-rho)*model_std**2)

        n_frame +=1

    return foreground


def single_gaussian_model(video_path, alpha, rho, adaptive=False):
    print('Computing Gaussian model...')
    mean, std = get_pixels_single_gaussian_model(video_path)
    print('Gaussian computed for pixels')
    print('Extracting Background...')
    bg = get_fg_mask_single_gaussian_model(video_path, first_frame=int(2141 * 0.25), model_mean=mean, model_std=std,
                                            alpha=alpha, rho=rho, adaptive=adaptive)
    print('Extracted background with shape {}'.format(bg.shape))


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

    while capture.isOpened():
        valid, frame = capture.read()
        if not valid:
            break
        images.append(frame)

        # Algorithms
        BackgroundSubtractorMOG(fgbg_MOG, frame, i, export_frames)
        BackgroundSubtractorMOG2(fgbg_MOG2, frame, i, export_frames)
        BackgroundSubtractorGMG(fgbg_GMG, frame, i, export_frames)

        i += 1

    capture.release()
    cv2.destroyAllWindows()

def BackgroundSubtractorMOG(fgbg_MOG, frame, i, export_frames=False):
    fgmask = fgbg_MOG.apply(frame)
    fgmask = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
    if export_frames:
        cv2.imwrite('output_frames/MOG/frame_{:04d}.jpg'.format(i), fgmask)

def BackgroundSubtractorMOG2(fgbg_MOG2, frame, i, export_frames=False):
    fgmask = fgbg_MOG2.apply(frame)
    fgmask = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
    if export_frames:
        cv2.imwrite('output_frames/MOG2/frame_{:04d}.jpg'.format(i), fgmask)

def BackgroundSubtractorGMG(fgbg_GMG, frame, i, export_frames=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    fgmask = fgbg_GMG.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.resize(fgmask, (0, 0), fx=0.3, fy=0.3)
    if export_frames:
        cv2.imwrite('output_frames/GMG/frame_{:04d}.jpg'.format(i), fgmask)
