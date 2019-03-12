import cv2
import os

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

############################################
########### State-of-the-art methods
############################################

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
