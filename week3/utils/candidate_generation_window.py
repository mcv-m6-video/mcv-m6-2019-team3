import cv2
from tqdm import tqdm

from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from utils.detection import Detection


def candidate_generation_window_ccl(n_frame, mask, min_h=80, max_h=500, min_w=100, max_w=600, min_ratio=0.2, max_ratio=1.30):
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
    fig, ax = plt.subplots()
    ax.imshow(pixel_candidates, cmap='gray')

    for candidate in detection_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    for candidate in gt_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc-minc+1, maxr-minr+1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

    #plt.show()

def visualize_boxes_opencv(image, gt_candidate, detection_candidate):

    for candidate in detection_candidate:
        minc, minr, maxc, maxr = candidate
        cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 0, 255), 5)    # Red

    for candidate in gt_candidate:
        minc, minr, maxc, maxr = candidate
        cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 0), 3)    # Green

    return image

def plot_bboxes(video_path, groundtruth, detections, export_frames=False):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    pbar = tqdm(total=2140)

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break

        if n_frame >= 0: # and n_frame < 700:
            # Get groundtruth of the target frame
            gt_on_frame = [x for x in groundtruth if x.frame == n_frame]
            gt_bboxes = [o.bbox for o in gt_on_frame]
            detections_on_frame = [x for x in detections if x.frame == n_frame]
            detections_bboxes = [o.bbox for o in detections_on_frame]

            # Plot image
            frame = visualize_boxes_opencv(image, gt_bboxes, detections_bboxes)

            if export_frames:
                new_image = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                cv2.imwrite('output frames/mask-rcnn/frame_{:04d}.jpg'.format(n_frame), new_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 4, int(cv2.IMWRITE_JPEG_QUALITY), 70])

            pbar.update(1)

        n_frame += 1

    pbar.close()