#from week1.utils.reading import read_annotations_file
import numpy as np
import cv2


def get_pixels_single_gaussian_model(video_path, last_frame=int(2141*0.25)):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened() and n_frame <= last_frame:
        valid, image = capture.read()
        if not valid:
            break
        if n_frame==0:
            gaussians = np.zeros((image.shape[0], image.shape[1], last_frame+1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gaussians[:, :, n_frame] = image

        # Get groundtruth and detections from frame n
        #gt_on_frame = [x for x in groundtruth_list if x.frame == n]
        #gt_bboxes = [o.bbox for o in gt_on_frame]

        n_frame += 1

    gauss_mean = gaussians.mean(axis=2)
    gauss_std = gaussians.std(axis=2)

    return gauss_mean, gauss_std


def get_frame_mask_single_gaussian_model(img, model_mean, model_std, alpha):
    foreground = (abs(img - model_mean) >= alpha*(model_std+2))

    return foreground


def get_fg_mask_single_gaussian_model(video_path, first_frame, model_mean, model_std, alpha, rho, adaptive=False):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        if n_frame == first_frame:
            foreground = np.zeros((image.shape[0], image.shape[1], 2141 - first_frame))
        if n_frame > first_frame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            foreground[:, :, n_frame-first_frame-1] = get_frame_mask_single_gaussian_model(image, model_mean, model_std, alpha)

            if adaptive:
                model_mean = rho*image + (1-rho)*model_mean
                model_std = np.sqrt(rho*(image - model_mean)**2 + (1-rho)*model_std**2)

        n_frame +=1

    return foreground


def single_gaussian_model(video_path, alpha, rho, adaptive=False):
    print('Computing Gaussian model...')
    mean, std = get_pixels_single_gaussian_model(video_path)
    print('Gaussian computed for pixels')
    print('Extracting Background...')
    bg = get_fg_mask_single_gaussian_model(video_path, first_frame=int(2141 * 0.25), model_mean=mean, model_std=std,
                                            alpha=alpha, rho=rho, adaptive=adaptive)
    print('Extracted background with shape {}'.format(bg.shape))


if __name__ == '__main__':

    video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
    groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"

    ## Get groundtruth
    #print("Getting groundtruth")
    #groundtruth_list = read_annotations_file(groundtruth_path)          # gt.txt

    single_gaussian_model(video_path, alpha=2.5, rho=1, adaptive=True)




