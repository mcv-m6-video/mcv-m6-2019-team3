from utils.detection import Detection
import random


def add_noise(detection):
    w = detection.width
    h = detection.height
    xtl = detection.xtl
    ytl = detection.ytl

    w_translation = w*0.2
    h_translation = h*0.2
    rescale_min = 0.1
    rescale_max = 2

    xtl_2 = min(round(xtl + random.uniform(0, 1)*h_translation), 1920)
    ytl_2 = min(round(ytl + random.uniform(0, 1)*w_translation), 1080)
    w_2 = round(w*random.uniform(rescale_min, rescale_max))
    h_2 = round(h*random.uniform(rescale_min, rescale_max))

    return Detection(detection.frame, detection.label, xtl_2, ytl_2, w_2, h_2, None)


def generate_detection(frame_id, min_prob = 0.5):
    if random.uniform(0, 1) > min_prob:
        xtl = round(random.uniform(0, 1910))
        ytl = round(random.uniform(0, 1070))
        return Detection(frame=frame_id, label='car', xtl = xtl, ytl = ytl, width=round(min(random.uniform(50, 300), 1920-xtl)), height=round(min(random.uniform(50, 200), 1080-ytl)), confidence=None)
    else:
        return


def delete_detection(min_prob = 0.8):
    if random.uniform(0, 1) > min_prob:
        return True
    return False


def obtain_modified_detections(detections):
    frame_detections = []

    for detection in detections:
        if not delete_detection():
            frame_detections.append(detection)
        for i in range(random.randint(1, 6)):
            frame_detections.append(add_noise(detection))
        new_detection = generate_detection(detection.frame)
        if new_detection:
            frame_detections.append(new_detection)

    return frame_detections





