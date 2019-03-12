from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def candidate_generation_window_ccl(mask):
    label_image = label(mask)
    regions = regionprops(label_image)

    window_candidates = []
    for region in regions:
        bbox = list(region.bbox)
        box_h, box_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        window_candidates.append(bbox)

    return window_candidates


def visualize_boxes(pixel_candidates, window_candidates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pixel_candidates)
    for candidate in window_candidates:
        minr, minc, maxr, maxc = candidate
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()