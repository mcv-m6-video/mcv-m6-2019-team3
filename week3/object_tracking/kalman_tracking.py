from object_tracking.sort import Sort, associate_detections_to_trackers
import cv2
from utils.plotting import visualize_tracks, visualize_tracks_opencv
from utils.track import Track
from object_tracking.tracking import predict_position
import numpy as np
import motmetrics as mm
from tqdm import tqdm
from utils.detection import Detection

# def read_detections(detections, min_confidence=0.6, classes_of_interest=['car', 'bus', 'truck']):
#     bounding_boxes, scores, classes = detections
#     detections = []
#     for box, score, category in zip(bounding_boxes, scores, classes):
#         if score >= min_confidence and category in classes_of_interest:
#             detections.append(box)
#     return np.array(detections)

def kalman_track_objects(video_path, detections_list, gt_list, display=False, export_frames=False):
    capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    tracks = []
    pbar = tqdm(total=2140)
    acc = mm.MOTAccumulator(auto_id=True)
    new_detections = []

    # detections_list = read_detections(detections_list)
    kalman_tracker = Sort()
    while capture.isOpened():
        valid, frame = capture.read()
        if not valid:
            break
        frame_idx += 1
        unused_detections = detections_list
        frame_tracks = {}
        detections_on_frame = [x for x in detections_list if x.frame == frame_idx]
        gt_on_frame = [x for x in gt_list if x.frame == frame_idx]

        detections_formatted = [[x.bbox[0], x.bbox[1], x.bbox[2], x.bbox[3], x.confidence] for x in detections_on_frame]
        gt_formatted = [[x.bbox[0], x.bbox[1], x.bbox[2], x.bbox[3], x.confidence] for x in gt_on_frame]        
        trackers = kalman_tracker.update(detections_formatted)
        pbar.update(1)
        new_detections.append(trackers)
        # IDF1 computing
        detec_bboxes = []
        detec_ids = []
        gt_bboxes = []
        gt_ids = []
        for gt in gt_on_frame:
            gt_bboxes.append(gt.bbox)
            gt_ids.append(gt.track_id)
        # new_detections.append(for line in trackers)
        for track_det in trackers:
        #     updatedDetection = Detection(frame_idx, )
            detec_bboxes.append(track_det[:4])
            detec_ids.append(track_det[4])
        mm_gt_bboxes = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in gt_bboxes]
        mm_detec_bboxes = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in detec_bboxes]
        distances_gt_det = mm.distances.iou_matrix(mm_gt_bboxes, mm_detec_bboxes, max_iou=1.)
        acc.update(gt_ids, detec_ids, distances_gt_det)
        if display:
            for track_det in trackers:
                track_det = track_det.astype(np.uint32)
                font = cv2.FONT_ITALIC
                placement = (track_det[2] + 10, track_det[3] + 10)
                
                cv2.rectangle(frame, (track_det[0], track_det[1]), (track_det[2], track_det[3]), (0, 0, 255), 3)
                font_scale = 1
                font_color = (200, 200, 0)
                line_type = 2
                cv2.putText(frame, str(track_det[4]), placement, font, font_scale, font_color, line_type)
            cv2.imshow('output', frame)
            cv2.waitKey()
    
    pbar.close()
    capture.release()
    cv2.destroyAllWindows()
    print(acc.mot_events)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary)
    return new_detections

            
        

