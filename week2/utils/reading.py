import cv2
import xml.etree.ElementTree as ET

from week2.utils.detection import Detection


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

            frame_detections.append(Detection(frame_id, 'car', tl_x, tl_y, width, height, 1))

    return frame_detections


def read_annotations(annotation_path):
    """
    Arguments: 
    capture: frames from video, opened as cv2.VideoCapture
    root: parsed xml annotations as ET.parse(annotation_path).getroot()
    """
    capture = cv2.VideoCapture("./datasets/AICity_data/train/S03/c010/vdo.avi")
    root = ET.parse(annotation_path).getroot()

    ground_truths = []
    images = []
    num = 0
    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        #for now: (take only numannotated annotated frames)
        #if num > numannotated:
        #    break

        images.append(image)
        for track in root.findall('track'):
            #gt_id = track.attrib['id']
            label = track.attrib['label']
            box = track.find("box[@frame='{0}']".format(str(num)))
            if box is not None and label == 'car':
                frame = int(box.attrib['frame'])
                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))
                #ground_truths.append(Detection(frame, label, xtl, ytl, xbr - xtl + 1, ybr - ytl + 1, 1))
                ground_truths.append(Detection(frame, label, xtl, ytl, xbr, ybr, 1))

        num += 1

    # print(ground_truths)
    capture.release()
    return ground_truths


def read_annotations_from_txt(gt_path):
    """
    Read annotations from the txt files
    Arguments:
    gt_path: path to .txt file
    :returns: list of Detection
    """
    ground_truths_list = list()
    with open(gt_path) as f:
        for line in f:
            data = line.split(',')
            ground_truths_list.append(Detection(int(data[0]), 'car', int(float(data[2])), int(float(data[3])), int(float(data[2])) + int(float(data[4])), int(float(data[3])) + int(float(data[5])),float(data[6])))

    return ground_truths_list

def read_annotations_file(gt_path):
    if (gt_path.endswith('.txt')):
        annotations_list = read_annotations_from_txt(gt_path)
    elif (gt_path.endswith('.xml')):
        annotations_list = read_annotations(gt_path)
    else:
        raise Exception('Incompatible filetype')

    return annotations_list
