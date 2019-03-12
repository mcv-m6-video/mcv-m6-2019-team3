from processing.background_subtraction import BackgroundSubtractor

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"

if __name__ == "__main__":

    # State-of-the-art background subtractors
    BackgroundSubtractor(video_path, export_frames=True)
