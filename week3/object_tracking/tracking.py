import cv2

from utils.track import Track
from evaluation.bbox_iou import bbox_iou


def obtain_new_tracks(tracks, unused_bboxes, max_track):
    for bbox in unused_bboxes:
        tracks.append(Track(max_track+1, [bbox], 0, 1, 1))
        max_track += 1

    return tracks, max_track


def update_tracks(tracks, detections_bboxes):
    unused_detections_bboxes = detections_bboxes
    for track in tracks:
        if track.time_since_update < 5:
            last_bbox = track.bboxes[-1]
            match_bbox = match_next_bbox(last_bbox, unused_detections_bboxes)
            if match_bbox is not None:
                unused_detections_bboxes.remove(match_bbox)
                track.bboxes.append(match_bbox)
                track.hits +=1
                if track.time_since_update == 0:
                    track.hit_streak += 1
            else:
                track.time_since_update += 1
    return tracks, unused_detections_bboxes


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


def track_objects(video_path, detections_list):

    tracks = []
    max_track = 0

    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break

        detections_on_frame = [x for x in detections_list if x.frame == n_frame]
        detections_bboxes = [o.bbox for o in detections_on_frame]

        tracks, unused_bboxes = update_tracks(tracks, detections_bboxes)
        tracks, max_track = obtain_new_tracks(tracks, unused_bboxes, max_track)

        n_frame += 1
    capture.release()
    cv2.destroyAllWindows()

    return tracks