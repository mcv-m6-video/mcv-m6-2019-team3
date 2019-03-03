from evaluation.evaluation_funcs import performance_accumulation_pixel, performance_accumulation_window, performance_evaluation_pixel, performance_evaluation_window
import numpy as np
import xml.etree.ElementTree as ET
from typing import Iterator, Tuple
import cv2
from utils.detection import Detection


video_path = "./datasets/AICity_data/train/S03/c010/vdo.avi"
annotation_path = "./datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml"

if __name__ == "__main__":
    capture = cv2.VideoCapture(video_path)
    root = ET.parse(annotation_path).getroot()

    ground_truths = []
    ground_truth_ids = []
    num = 0
    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        #for now: (take only 40 annotated frames)
        if num > 40:
            break

        for track in root.findall('track'):
            gt_id = track.attrib['id']
            label = track.attrib['label']
            box = track.find("box[@frame='{0}']".format(str(num)))
            if box is not None:
                print(box)
                print(gt_id)

                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))
                ground_truth_ids.append(gt_id)
                ground_truths.append(Detection(gt_id, label, xtl, ytl, xbr - xtl + 1, ybr - ytl + 1))
        num += 1

    print(ground_truths)
    capture.release()



def read_detections(path: str):
    # [frame, -1, left, top, width, height, conf, -1, -1, -1]
    frame_detections = []

    with open(path) as f:
        for line in f.readlines():
            parts = line.split(',')
            frame_id = int(parts[0])
            # while frame_id > len(frame_detections):
            #     frame_detections.append([])

            tl_x = int(float(parts[2]))
            tl_y = int(float(parts[3]))
            width = int(float(parts[4]))
            height = int(float(parts[5]))

            frame_detections.append({frame_id, 'car', tl_x, tl_y, width, height})

    return frame_detections
