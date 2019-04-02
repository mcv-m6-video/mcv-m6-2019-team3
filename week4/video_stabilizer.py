import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

from block_matching import block_matching_optical_flow
from utils.trajectory import Trajectory

video_path = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\piano.mp4'
sequences_path = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\imgs\\'
save_path = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\stab\\'

save_out_path = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\piano_out_pfm\\'
save_in_path = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\piano_in\\'
save_out_simple = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\piano_out_simple\\'
save_out = 'C:\\Users\\Usuario\\Desktop\\Temporal Sara\\mcv-m6-2019-team3\\datasets\\cat_stab\\piano_out\\'

SMOOTHING_RADIUS = 5  #In frames. The larger the more stable the video, but less reactive to sudden panning


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def point_feature_matching(vpath=video_path):
    # Read input video
    cap = cv2.VideoCapture(vpath)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Set up output video
    #out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in range(n_frames-2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                         maxCorners=200,
                                         qualityLevel=0.01,
                                         minDistance=30,
                                         blockSize=3)
        # Read next frame
        success, curr = cap.read()
        if not success:
            break
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        #Find transformation matrix
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])

        # Store transformation
        transforms[i] = [dx,dy,da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        cv2.imwrite(save_out_path + str(i) + '.png', frame_stabilized)
        #cv2.imwrite(save_in_path + str(i) + '.png', frame)
        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        if(frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2))

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        #out.write(frame_out)

    # Release video
    cap.release()
    #out.release()
    # Close windows
    cv2.destroyAllWindows()


def video_stabilization(sequence, GT = None):

    N = len(sequence)
    # N = 10

    prev = sequence[0]
    H, W, C = prev.shape

    seq_stabilized = np.zeros((H, W, C, N-1))
    seq_stabilized[:, :, :, 0] = prev

    traj_a = 0
    traj_x = 0
    traj_y = 0

    trajs = []

    origs = []

    # 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    for i in range(1, N-1):

        next = sequence[i]

        vector_field = block_matching_optical_flow(prev, next)

        mag, ang = cv2.cartToPolar(vector_field[0], vector_field[1])
        mags, times = np.unique(mag, return_counts=True)
        magf = mags[times.argmax()]
        angs, times = np.unique(ang, return_counts=True)
        da = angs[times.argmax()]

        dx = magf*np.cos(da)
        dy = magf*np.sin(da)

        H = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        next_stabilized = cv2.warpAffine(next, H, (next.shape[1], next.shape[0]))  # translation + rotation only

        cv2.imwrite(save_out_simple + str(i) + '.png', next_stabilized)

    # 2 - Accumulate the transformations to get the image trajectory
        traj_x = traj_x + dx
        traj_y = traj_y + dy
        traj_a = traj_a + da

        trajs.append(Trajectory(traj_x, traj_y, traj_a))
        origs.append(Trajectory(dx, dy, da))
        prev = next_stabilized
        print("Trajectory {}: {}". format(i, Trajectory(traj_x, traj_y, traj_a)))

    plot_traj([item.x for item in origs], [item.y for item in origs], np.arange(N-2), title2="first")
    plot_traj([item.x for item in trajs], [item.y for item in trajs], np.arange(N-2), title2="trajs")

    smooth_trajs = []
    #  3 - Smooth out the trajectory using an averaging window
    for i in range(len(trajs)):
        sum_a = 0
        sum_x = 0
        sum_y = 0
        cnt = 0
        for j in range(-SMOOTHING_RADIUS, SMOOTHING_RADIUS+1):
            if 0 <= (i+j) < len(trajs):

                sum_x = sum_x + trajs[i+j].x
                sum_y = sum_y + trajs[i+j].y
                sum_a = sum_a + trajs[i+j].a
                cnt = cnt + 1
        avg_a = sum_a / cnt
        avg_x = sum_x / cnt
        avg_y = sum_y / cnt

        smooth_traj = Trajectory(avg_x, avg_y, avg_a)
        smooth_trajs.append(smooth_traj)

        print("Smooth Trajectory {}: {}". format(i, smooth_traj))

    plot_traj([item.x for item in smooth_trajs], [item.y for item in smooth_trajs], np.arange(N-2), title2="smooth")

    # 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    a = 0
    x = 0
    y = 0
    news = []
    for i, orig in enumerate(origs):
        x = orig.x + x
        y = orig.y + y
        a = orig.a + a

        diff_x = smooth_trajs[i].x - trajs[i].x
        diff_y = smooth_trajs[i].y - trajs[i].y
        diff_a = smooth_trajs[i].a - trajs[i].a

        dx = orig.x + diff_x
        dy = orig.y + diff_y
        da = orig.a + diff_a

        news.append(Trajectory(dx, dy, da))
        print("Final Trajectory {}: {}". format(i, Trajectory(dx, dy, da)))
    plot_traj([item.x for item in news], [item.y for item in news], np.arange(N-2), title2="final")

    # 5 - Apply the new transformation to the video
    for i in range(len(news)):
        next = sequence[i]
        # H = np.array([[np.cos(news[i].a), - np.sin(news[i].a), -news[i].x], [np.sin(news[i].a), np.cos(news[i].a), -news[i].y]],
        #            dtype=np.float32)
        H = np.array([[1, 0, -news[i].x], [0, 1, -news[i].y]],
                     dtype=np.float32)
        next_stabilized = cv2.warpAffine(next, H, (next.shape[1], next.shape[0]))
        cv2.imwrite(save_out + str(i) + '.png', next_stabilized)
        seq_stabilized[:, :, :, i] = next_stabilized

    return seq_stabilized


def read_sequences(path):
    imgNames = os.listdir(path)
    imgNames.sort()
    seq_images = []
    for name in imgNames:
            im = cv2.imread(path + name)
            seq_images.append(im)
    return seq_images


def plot_traj(dx, dy, frames, title="trajectory", title2=""):

    # Data for plotting
    fig, ax = plt.subplots()
    ax.plot(frames, dx)

    ax.set(xlabel='Frame number', ylabel='pixels',
           title='Trajectory X')

    ax.grid()

    fig.savefig(save_path + "\\plots\\vidstab-" + title + title2 + "X.png")
    plt.show()

    # Data for plotting
    fig2, ax2 = plt.subplots()
    ax2.plot(frames, dy)

    ax2.set(xlabel='Frame number', ylabel='pixels',
           title='Trajectory Y')
    ax2.grid()

    fig2.savefig(save_path + "\\plots\\vidstab-" + title + title2 + "Y.png")
    plt.show()


if __name__ == "__main__":

    print("Point Feature Matching")
    point_feature_matching(video_path)

    print("Video Stabilization with block matching")
    sequences = read_sequences(save_in_path)
    seq_stabilized = video_stabilization(sequences)




