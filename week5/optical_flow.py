import cv2
import numpy as np

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
