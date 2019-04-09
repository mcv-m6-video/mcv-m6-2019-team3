import os
import cv2
import numpy as np
from numpy.linalg import inv
from collections import defaultdict, Counter

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.plotting import visualize_tracks, visualize_tracks_opencv
from evaluation.bbox_iou import bbox_iou
from utils.reading import read_homography_matrix


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


def bboxes_correspondences(gt_detections, timestamps, framenum, fps):
    correspondences = []

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in gt_detections[0] if detec.frame == frame]

        for cam in range(1, len(framenum)):
            if frame >= int(timestamps[cam]*fps) and frame <= framenum[cam]:
                detecs_frame_1 = [detec for detec in gt_detections[1] if detec.frame == frame - int(timestamps[cam]*fps)]

                for detec_0 in detecs_frame_0:
                    matches = [detec_1 for detec_1 in detecs_frame_1 if detec_1.track_id == detec_0.track_id]
                    if len(matches) == 1:

                        correspondences.append((detec_0.bbox, matches[0].bbox))

    return correspondences


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


def apply_homography_to_point(x, y, H1, H2):
    pos_1 = np.array([x, y, 1])
    det_0_hom = inv(H1).dot(pos_1)
    det_0_hom = np.array([det_0_hom[0] / det_0_hom[2], det_0_hom[1] / det_0_hom[2], 1])
    det_1_hom = H2.dot(det_0_hom)

    return [det_1_hom[0]/det_1_hom[2], det_1_hom[1]/det_1_hom[2], 1]


def get_homography_IoU(det_0, det_1, homography1, homography2):

    minc, minr, maxc, maxr = det_0.bbox
    H1 = np.array(homography1)
    H2 = np.array(homography2)

    x = minc
    y = maxr
    det_1_hom = apply_homography_to_point(x, y, H1, H2)

    x_br = maxc
    y_br = maxr
    det_1_hom_br = apply_homography_to_point(x_br, y_br, H1, H2)


    original_ratio = (maxr - minr)/(maxc - minc) #height/width of bbox
    width_transformed = abs(det_1_hom[0] - det_1_hom_br[0])
    height_transformed = original_ratio*width_transformed

    predicted_bbox_1 = [min(det_1_hom[0],det_1_hom_br[0]), min(det_1_hom[1],det_1_hom_br[1])-height_transformed, max(det_1_hom[0],det_1_hom_br[0]), max(det_1_hom[1],det_1_hom_br[1])]

    IoU = bbox_iou(det_1.bbox, predicted_bbox_1)

    return IoU

def transform_detection(det_0, homography1):
    minc, minr, maxc, maxr = det_0.bbox
    H1 = np.array(homography1)
    H2 = np.array(homography2)

    x = minc
    y = maxr
    det_1_hom = apply_homography_to_point(x, y, H1, H2)

    x_br = maxc
    y_br = maxr
    det_1_hom_br = apply_homography_to_point(x_br, y_br, H1, H2)

    original_ratio = (maxr - minr) / (maxc - minc)  # height/width of bbox
    width_transformed = abs(det_1_hom[0] - det_1_hom_br[0])
    height_transformed = original_ratio * width_transformed

    predicted_bbox_1 = [min(det_1_hom[0], det_1_hom_br[0]), min(det_1_hom[1], det_1_hom_br[1]) - height_transformed,
                        max(det_1_hom[0], det_1_hom_br[0]), max(det_1_hom[1], det_1_hom_br[1])]

    return predicted_bbox_1

def match_correspondence(detection, correspondences):
    highest_IoU = 0
    for correspondence in correspondences:
        IoU = bbox_iou(correspondence[0], detection.bbox)

        if IoU > highest_IoU:
            highest_IoU = IoU
            best_match = correspondence

    if highest_IoU > 0:
        return best_match
    else:
        return None


def get_correspondence_IoU(det_0, det_1, correspondences):
    closer_det = match_correspondence(det_0, correspondences)
    if closer_det:
        closer_det_1 = closer_det[1]
        IoU = bbox_iou(closer_det_1, det_1.bbox)
        return IoU
    return 0


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
    colors[499] = [1, 0, 0]
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
                else:
                    frame_tracks_0[499] = dict(bbox=detec.bbox)

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
                if detec.track_id in matched_tracks.values():
                    frame_tracks_1[detec.track_id] = dict(bbox=detec.bbox)
                else:
                    frame_tracks_1[499] = dict(bbox=detec.bbox)

        visualize_tracks_opencv(image, frame_tracks_1, colors, export_frames=True,
                                export_path="output_frames/c002/frame_{:04d}.png".format(n_frame))
        if n_frame%2 == 0:
            visualize_tracks(image, frame_tracks_1, colors, display=False)
        n_frame += 1


def match_tracks(cameras_tracks, homographies, timestamps, framenum, fps, video_path_0, video_path_1):

    for camera1 in cameras_tracks:
        for camera2 in cameras_tracks:
            if camera2 != camera1:
                tracks_camera1 = cameras_tracks[camera1]
                tracks_camera2 = cameras_tracks[camera2]
                for track1 in tracks_camera1:
                    transformed_track = []
                    for detec in track1.detections:
                        transformed_track.append(transform_detection(detec, homographies[camera1]))



def match_tracks_by_frame(cameras_tracks, homographies, timestamps, framenum, fps, video_path_0, video_path_1, correspondences):
    dist = defaultdict(list)

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in cameras_tracks[0] if detec.frame == frame]
        #detecs_3d_0 = compute_3d_detecs(detecs_frame_0, homographies[0])
        for cam in range(1, len(framenum)):
            for frame2 in range(frame - int(timestamps[cam]*fps)-5, frame - int(timestamps[cam]*fps) + 6):
                if frame2 >= 0 and frame2 <= framenum[cam]:
                    detecs_frame_1 = [detec for detec in cameras_tracks[1] if detec.frame == frame2]
                    #detecs_3d_1 = compute_3d_detecs(detecs_frame_1, homographies[cam])

                    for det_0 in detecs_frame_0:
                        for det_1 in detecs_frame_1:
                            dist[det_0.track_id].append({det_1.track_id: (intersection(det_0.histogram, det_1.histogram), get_homography_IoU(det_0, det_1, homographies[0], homographies[1]))}) #dict of tuples (hist intersection, correspondence IoU)

    total_distances = []
    for key, dist_list in dist.items():
        for item in dist_list:
            d = tuple(item.values())[0]
            total_distances.append(float(d[0])) #histograms


    print(total_distances)
    max_dist = max(total_distances)
    min_dist = min(total_distances)
    average = float(sum(total_distances)) / len(total_distances)
    print('max dist: {}'.format(max_dist))
    print('min dist: {}'.format(min_dist))
    print('average dist: {}'.format(average))

    threshold = average

    candidates = defaultdict(list)
    for track_0, dist_list in dist.items():
        for track_1_key in dist_list:
            k = list(track_1_key.keys())
            track_1 = k[0]
            d = tuple(track_1_key.values())[0]
            if float(d[0]) < threshold:
                candidates[track_0].append(track_1)

    match_tracks = {}
    for track_0, dist_list in dist.items():
        possible_tracks = []
        max_IoU = 0
        cnt = Counter()
        for track_1_key in dist_list:
            k = list(track_1_key.keys())
            track_1 = k[0]
            if track_1 in candidates[track_0]:
                possible_tracks.append(track_1)
                d = tuple(track_1_key.values())[0]
                if float(d[1]) > max_IoU:
                    max_IoU = float(d[1])
                    best_track = track_1
        # if max_IoU == 1:
        #     cnt[best_track] += 1
        for track_pos in possible_tracks:
            if max_IoU > 0.30:
                if track_pos == best_track:
                    cnt[track_pos] += 1
            # else:
            #     cnt[track_pos] += 1
        #else:
        #    for track_pos in possible_tracks:
        #        cnt[track_pos] += 1
        match_tracks[track_0] =cnt

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

#
# if __name__ == '__main__':
#     x1, y1, z1 = lon_lat_to_cartesian(42.525821, -90.723853)
#     x2, y2, z2 = lon_lat_to_cartesian(42.525473, -90.723319)
#
#     print('camera1: {0}, {1}, {2}'.format(x1, y1, z1))
#     print('camera2: {0}, {1}, {2}'.format(x2, y2, z2))
#


if __name__ == '__main__':
    frame_path_1 = '../video_frames/c001/frame_0357.png'
    bbox_1 = [474,496,(474+651),(496+290)]
    print(bbox_1)
    frame_path_2 = '../video_frames/c002/frame_0338.png'
    homography_path = "calibration.txt"
    camera1 = "../../datasets/AICity_data/train/S01/c001/"
    camera2 = "../../datasets/AICity_data/train/S01/c002/"
    homography_path_start = "../../datasets/calibration/"

    homography1 = read_homography_matrix(homography_path_start + camera1[(len(camera1)-5):] + homography_path)
    homography2 = read_homography_matrix(homography_path_start + camera2[(len(camera2)-5):] + homography_path)

    image1 = cv2.imread(frame_path_1)
    fig, ax = plt.subplots()
    ax.imshow(image1)

    minc, minr, maxc, maxr = bbox_1
    rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='green', linewidth=4)
    ax.add_patch(rect)
    plt.show()

    H1 = np.array(homography1)
    H2 = np.array(homography2)
    minc, minr, maxc, maxr = bbox_1
    h, w = image1.shape[:2]
    x = minc
    y = maxr

    det_1_hom = apply_homography_to_point(x, y, H1, H2)

    x_br = maxc
    y_br = maxr
    det_1_hom_br = apply_homography_to_point(x_br, y_br, H1, H2)

    print('Left: {}'.format(det_1_hom))
    print('Right: {}'.format(det_1_hom_br))
    print('')


    original_ratio = (maxr - minr)/(maxc - minc) #height/width of bbox
    width_transformed = abs(det_1_hom[0] - det_1_hom_br[0])
    height_transformed = original_ratio*width_transformed

    predicted_bbox_1 = [min(det_1_hom[0],det_1_hom_br[0]), min(det_1_hom[1],det_1_hom_br[1])-height_transformed, max(det_1_hom[0],det_1_hom_br[0]), max(det_1_hom[1],det_1_hom_br[1])]


    image2 = cv2.imread(frame_path_2)
    h,w = image2.shape[:2]
    fig, ax = plt.subplots()
    ax.imshow(image2)
    gt_bbox = [1163,420,(1163+524),(420+245)]
    predicted_bbox_1 = [predicted_bbox_1[0], predicted_bbox_1[1], predicted_bbox_1[2], predicted_bbox_1[3]]
    print('Predicted bbox: {}'.format(predicted_bbox_1))
    print('Ground truth bbox: {}'.format(gt_bbox))
    minc, minr, maxc, maxr = gt_bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='green',
                              linewidth=4)
    ax.add_patch(rect)
    minc, minr, maxc, maxr = predicted_bbox_1
    rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='red',
                              linewidth=4)
    ax.add_patch(rect)
    plt.show()



