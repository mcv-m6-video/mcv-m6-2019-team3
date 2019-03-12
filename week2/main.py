from processing.background_subtraction import BackgroundSubtractor, single_gaussian_model

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"

if __name__ == "__main__":

    # Gaussian modelling
    single_gaussian_model(video_path, alpha=2.5, rho=1, adaptive=True, export_frames=True)

    # State-of-the-art background subtractors
    #BackgroundSubtractor(video_path, export_frames=False)
