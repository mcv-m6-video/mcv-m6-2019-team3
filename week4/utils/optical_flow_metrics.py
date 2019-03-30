"""Optical flow maps are saved as 3-channel uint16 PNG images: 
The first channel contains the u-component
The second channel the v-component
The third channel denotes if a valid ground truth optical flow value exists for that
pixel (1 if true, 0 otherwise). 

To convert the u-/v-flow into floating point values:
-convert the value to float
-subtract 2^15 
-divide the result by 64:

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_flow_data(gt_noc, test):
    img_test = cv2.imread(test, -1)
    img_gt = cv2.imread(gt_noc, -1)

    if (img_test.shape[0] != img_gt.shape[0]) or (img_test.shape[1] != img_gt.shape[1]):
        print("ERROR: Wrong file size!")
        return
    else:

        fu_test = (img_test[:, :, 2] - 2. ** 15) / 64
        fv_test = (img_test[:, :, 1] - 2. ** 15) / 64
        valid_test = img_test[:,:,0]

        fu_gt = (img_gt[:, :, 2] - 2. ** 15) / 64
        fv_gt = (img_gt[:, :, 1] - 2. ** 15) / 64
        valid_gt = img_gt[:,:,0]

        flow_gt = np.transpose(np.array([fu_gt, fv_gt, valid_gt]))

        flow_test = np.transpose(np.array([fu_test, fv_test, valid_test]))
        print(flow_test[:-1].shape)
        plt.figure(1)
        plt.imshow(flow_gt)
        plt.savefig('results/gt_flow.png')
        plt.show()

        plt.figure(2)
        plt.imshow(flow_test)
        plt.savefig('results/test_flow.png')
        plt.show()
        
        return flow_gt, flow_test

def compute_msen(flow_gt, flow_test, threshold=3):

    flow_u = flow_gt[:,:,0] - flow_test[:,:,0]
    flow_v = flow_gt[:,:,1] - flow_test[:,:,1]
    flow_err = np.sqrt(flow_u*flow_u+flow_v*flow_v)

    valid_gt = flow_gt[:,:, 2]

    flow_err[valid_gt == 0] = 0 

    plt.figure(1)
    plt.imshow(np.transpose(flow_err), cmap="jet")
    plt.colorbar()
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.savefig('results/err_flow.png')
    plt.show()

    msen = np.mean(flow_err[valid_gt != 0])

    pepn = (np.sum(flow_err[valid_gt != 0] > threshold)/len(flow_err[valid_gt != 0]))*100

    plt.figure(2)
    plt.hist(flow_err[valid_gt == 1], bins=50, density=True)
    plt.title('Optical Flow error')
    plt.xlabel('msen')
    plt.ylabel('Percentage of pixels')
    plt.savefig('results/msen.png')
    plt.show()
    
   

    return msen, pepn



if __name__ == "__main__":

    test = "../../datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"
    gt_noc = "../../datasets/data_stereo_flow/training/flow_noc/000045_10.png"

    flow_gt, flow_test = read_flow_data(gt_noc, test)
    
    msen, pepn = compute_msen(flow_gt, flow_test)

    print("MSEN: {}".format(msen))
    print("PEPN: {}".format(pepn))

