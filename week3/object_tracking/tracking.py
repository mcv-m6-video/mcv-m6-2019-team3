import cv2
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from utils.track import Track
from evaluation.bbox_iou import bbox_iou


def obtain_new_tracks(tracks, unused_bboxes, max_track, frame_tracks):
    for bbox in unused_bboxes:
        tracks.append(Track(max_track+1, [bbox], 0, 1, 1))
        frame_tracks[max_track+1] = bbox

        max_track += 1

    return tracks, max_track, frame_tracks


def update_tracks(tracks, detections_bboxes, frame_tracks):
    unused_detections_bboxes = detections_bboxes
    for track in tracks:
        if track.time_since_update < 5:
            last_bbox = track.bboxes[-1]
            match_bbox = match_next_bbox(last_bbox, unused_detections_bboxes)
            if match_bbox is not None:
                unused_detections_bboxes.remove(match_bbox)
                track.bboxes.append(match_bbox)
                frame_tracks[track.id] = match_bbox
                track.hits +=1
                if track.time_since_update == 0:
                    track.hit_streak += 1
            else:
                track.time_since_update += 1
    return tracks, unused_detections_bboxes, frame_tracks


def match_next_bbox(last_bbox, unused_detections_bboxes):
    highest_IoU = 0
    for detection_bbox in unused_detections_bboxes:
        IoU = bbox_iou(last_bbox, detection_bbox)
        if IoU > highest_IoU:
            highest_IoU = IoU
            best_match = detection_bbox

    if highest_IoU > 0:
        return best_match
    else:
        return None


def visualize_tracks(image, frame_tracks, colors):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    for id in frame_tracks.keys():
        bbox = frame_tracks[id]
        minc, minr, maxc, maxr = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor=colors[id],
                                  linewidth=2)
        ax.add_patch(rect)

    plt.show()



def track_objects(video_path, detections_list, display = True):
    colors = np.random.rand(100, 3)  # used only for display
    tracks = []
    max_track = 0

    capture = cv2.VideoCapture(video_path)
    n_frame = 0


    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        frame_tracks = {}

        detections_on_frame = [x for x in detections_list if x.frame == n_frame]
        detections_bboxes = [o.bbox for o in detections_on_frame]

        tracks, unused_bboxes, frame_tracks = update_tracks(tracks, detections_bboxes, frame_tracks)
        tracks, max_track, frame_tracks = obtain_new_tracks(tracks, unused_bboxes, max_track, frame_tracks)

        if display and n_frame%10==0:
            visualize_tracks(image, frame_tracks, colors)

        n_frame += 1
    capture.release()
    cv2.destroyAllWindows()

    return tracks