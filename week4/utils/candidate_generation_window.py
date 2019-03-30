from skimage.measure import label, regionprops
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
