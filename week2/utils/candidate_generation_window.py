import cv2

from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from week2.utils.detection import Detection


def candidate_generation_window_ccl(n_frame, mask):
    label_image = label(mask)
    regions = regionprops(label_image)

    window_candidates = []
    for region in regions:
        bbox = region.bbox
        box_h, box_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        det = Detection(frame=n_frame, label='car', xtl=bbox[1], ytl=bbox[0], width=box_w, height=box_h, confidence=1)
        window_candidates.append(det)

    return window_candidates


def visualize_boxes(pixel_candidates, gt_candidate, detection_candidate):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pixel_candidates)
    for candidate in gt_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc-minc+1, maxr-minr+1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

    for candidate in detection_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc-minc+1, maxr-minr+1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()

def plot_bboxes(video_path, groundtruth, detections):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break

        if n_frame > 900:
            # Get groundtruth of the target frame
            gt_on_frame = [x for x in groundtruth if x.frame == n_frame]
            gt_bboxes = [o.bbox for o in gt_on_frame]
            detections_on_frame = [x for x in detections if x.frame == n_frame]
            detections_bboxes = [o.bbox for o in detections_on_frame]
            visualize_boxes(image, gt_bboxes, detections_bboxes)

        n_frame += 1