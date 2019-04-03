import cv2
import numpy as np

import pyflow

def pyflow(frame1, frame2):

    u, v, im2W = pyflow.coarse2fine_flow(frame1, frame2, 0.012, 0.75, 20, 7, 1, 30, 1)
    return np.concatenate((u[..., None], v[..., None]), axis=2)

def farneback(frame1, frame2):

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, flow=None,
                                        pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)

    # Plot
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imwrite('opticalhsv.png', bgr)

    return bgr

def lucas_kanade(frame1, frame2):

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    p0 = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    #mask = np.zeros_like(frame1)                    # Create a mask image for drawing purposes
    #color = np.random.randint(0, 255, (100, 3))     # Create some random colors
    #for i, (new, old) in enumerate(zip(good_new, good_old)):
    #    a, b = new.ravel()
    #    c, d = old.ravel()
    #    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    #    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    #img = cv2.add(frame, mask)
    #cv2.imwrite('frame.png', img)

    mask = np.zeros_like(frame1)                    # Create a mask image for drawing purposes
