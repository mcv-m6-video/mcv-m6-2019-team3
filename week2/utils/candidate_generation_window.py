import cv2

from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from utils.detection import Detection


def candidate_generation_window_ccl(n_frame, mask, min_h, max_h, min_w, max_w, min_ratio, max_ratio):
    label_image = label(mask)
    regions = regionprops(label_image)

    window_candidates = []
    for region in regions:
        bbox = region.bbox
        if filter_connected_components(bbox, min_h, max_h, min_w, max_w, min_ratio, max_ratio):
            box_h, box_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
            det = Detection(frame=n_frame, label='car', xtl=bbox[1], ytl=bbox[0], width=box_w, height=box_h, confidence=1)
            window_candidates.append(det)

    return window_candidates


def filter_connected_components(bbox, min_h, max_h, min_w, max_w, min_ratio, max_ratio):
    box_h, box_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if box_h>max_h or box_w>max_w:
        return False
    if box_h<min_h or box_w<min_w:
        return False
    if (box_h/box_w)>max_ratio or (box_h/box_w)<min_ratio:
        return False
    else:
        return True


def visualize_boxes(pixel_candidates, gt_candidate, detection_candidate):
    #print('hey')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pixel_candidates)
    for candidate in gt_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc-minc+1, maxr-minr+1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

    for candidate in detection_candidate:
        minc, minr, maxc, maxr = candidate
        #print('Minc:{0} Minr: {1} Maxc;{2} Maxr:{3}'.format(minc, minr, maxc, maxr))
        rect = mpatches.Rectangle((minc, minr), maxc-minc+1, maxr-minr+1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()


def plot_bboxes(video_path, groundtruth, detections):
    print('print')
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break

        if n_frame > 600 and n_frame < 1200:
            # Get groundtruth of the target frame
            gt_on_frame = [x for x in groundtruth if x.frame == n_frame]
            gt_bboxes = [o.bbox for o in gt_on_frame]
            detections_on_frame = [x for x in detections if x.frame == n_frame]
            detections_bboxes = [o.bbox for o in detections_on_frame]
            if (len(gt_bboxes) != len(detections_bboxes)):
                print(n_frame)
                print(len(gt_bboxes))
                print(len(detections_bboxes))
            visualize_boxes(image, gt_bboxes, detections_bboxes)

        n_frame += 1