import cv2
import numpy as np
from numpy.linalg import inv
from collections import defaultdict, Counter

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import pandas as pd

from utils.plotting import visualize_tracks, visualize_tracks_opencv


def create_dataset(gt_detections, timestamps, framenum, fps):
    train_data = list()
    train_labels = list()

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in gt_detections[0] if detec.frame == frame]

        for cam in range(1, len(framenum)):
            if frame >= int(timestamps[cam]*fps) and frame <= framenum[cam]:
                detecs_frame_1 = [detec for detec in gt_detections[1] if detec.frame == frame - int(timestamps[cam]*fps)]

                for detec_0 in detecs_frame_0:
                    matches = [detec_1 for detec_1 in detecs_frame_1 if detec_1.track_id == detec_0.track_id]
                    if len(matches) == 1:
                        train_data.append(np.array(detec_0.bbox))
                        train_labels.append(np.array(matches[0].bbox))


    return np.vstack(train_data), np.vstack(train_labels)


def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=[4]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4))

    optimizer = RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def predict_bbox(train_data, train_labels):
    model = build_model()
    #result = model.predict(train_data)

    class PrintDot(Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 100000
    history = model.fit(
        train_data, train_labels, steps_per_epoch=int(len(train_data)*0.8/32),
        epochs=EPOCHS, validation_split=0.2, validation_steps=int(len(train_data)*0.2/32), verbose=0,
        callbacks=[PrintDot()])

    print(history.history.keys())
    print(history.history['loss'][-1])

    # plt.plot(history.history['epoch'], history.history['mean_absolute_error'])
    # plt.plot(history.history['epoch'], history.history['val_mean_absolute_error'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # plt.savefig('first_try.jpg')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig('loss_valloss_2.jpg')
    plt.close()



def intersection(u, v):
    """
    Compare histograms based on their intersection.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """
    return cv2.compareHist(np.array(u), np.array(v), cv2.HISTCMP_BHATTACHARYYA)


def compute_3d_detecs(detections, homography):
    detecs_3d = {}
    for detec in detections:
        minc, minr, maxc, maxr = detec.bbox
        x = (maxc + minc)/2
        y = maxr
        H = np.array(homography)
        Hinv = inv(H)
        pos_2d = np.array([x, y, 1])
        detecs_3d[detec.track_id] =dict(position=Hinv.dot(pos_2d), histogram=detec.histogram)

    return detecs_3d


def optimize_matching(match_tracks):
    final_matches = {}
    used = []
    for track_0 in match_tracks:
        for track_1 in match_tracks[track_0].most_common():
            if track_1[0] not in used:
                final_matches[track_0] = track_1[0]
                used.append(track_1[0])
                break

    print(final_matches)
    return final_matches


def visualize_matches(matched_tracks, cameras_tracks_0, cameras_tracks_1, video_path_0, video_path_1):
    colors = np.random.rand(500, 3)
    capture = cv2.VideoCapture(video_path_0)
    n_frame = 0

    while capture.isOpened():
        frame_tracks_0 = {}
        valid, image = capture.read()
        if not valid:
            break
        for detec in cameras_tracks_0:
            if detec.frame == n_frame:
                if matched_tracks.get(detec.track_id):
                    frame_tracks_0[matched_tracks[detec.track_id]] = dict(bbox=detec.bbox)

        visualize_tracks_opencv(image, frame_tracks_0, colors, export_frames=True,
                                export_path="output_frames/c001/frame_{:04d}.png".format(n_frame))
        if n_frame%2 == 0:
            visualize_tracks(image, frame_tracks_0, colors, display=False)
        n_frame += 1

    capture = cv2.VideoCapture(video_path_1)
    n_frame = 0

    while capture.isOpened():
        frame_tracks_1 = {}
        valid, image = capture.read()
        if not valid:
            break
        for detec in cameras_tracks_1:
            if detec.frame == n_frame:
                frame_tracks_1[detec.track_id] = dict(bbox=detec.bbox)
        visualize_tracks_opencv(image, frame_tracks_1, colors, export_frames=True,
                                export_path="output_frames/c002/frame_{:04d}.png".format(n_frame))
        if n_frame%2 == 0:
            visualize_tracks(image, frame_tracks_1, colors, display=False)
        n_frame += 1


def match_tracks(cameras_tracks, homographies, timestamps, framenum, fps, video_path_0, video_path_1):
    dist = defaultdict(list)

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in cameras_tracks[0] if detec.frame == frame]
        detecs_3d_0 = compute_3d_detecs(detecs_frame_0, homographies[0])

        for cam in range(1, len(framenum)):
            if frame >= int(timestamps[cam]*fps) and frame <= framenum[cam]:
                detecs_frame_1 = [detec for detec in cameras_tracks[1] if detec.frame == frame - int(timestamps[cam]*fps)]
                detecs_3d_1 = compute_3d_detecs(detecs_frame_1, homographies[cam])

        for track_0, det_0 in detecs_3d_0.items():
            for track_1, det_1 in detecs_3d_1.items():
                dist[track_0].append({track_1: intersection(det_0['histogram'], det_1['histogram'])})

    print(dist)
    total_distances = []
    for key, dist_list in dist.items():
        for item in dist_list:
            total_distances.append(float(max(item.values())))

    print(total_distances)
    max_dist = max(total_distances)
    min_dist = min(total_distances)
    average = float(sum(total_distances)) / len(total_distances)
    print('max dist: {}'.format(max_dist))
    print('min dist: {}'.format(min_dist))
    print('average dist: {}'.format(average))

    threshold = average

    match_tracks = {}

    for track_0, dist_list in dist.items():
        cnt = Counter()
        for list_item in dist_list:
            for i in list_item:
                if list_item[i] < threshold:
                    cnt[i] += 1
        match_tracks[track_0] = cnt

    print(match_tracks)

    matched_tracks = optimize_matching(match_tracks)

    visualize_matches(matched_tracks, cameras_tracks[0], cameras_tracks[1], video_path_0, video_path_1)


def lon_lat_to_cartesian(lon, lat, R = 6378137):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z


if __name__ == '__main__':
    x1, y1, z1 = lon_lat_to_cartesian(42.525821, -90.723853)
    x2, y2, z2 = lon_lat_to_cartesian(42.525473, -90.723319)

    print('camera1: {0}, {1}, {2}'.format(x1, y1, z1))
    print('camera2: {0}, {1}, {2}'.format(x2, y2, z2))


