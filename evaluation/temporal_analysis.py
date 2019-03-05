import matplotlib.pyplot as plt
import numpy as np
import os

def plotIoU(IoUFrames, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    IoUFrames = np.array(IoUFrames)
    plt.plot(IoUFrames[:,0])
    plt.ylabel("TP")
    plt.xlabel("#frame")
    plt.savefig(dir+"/TP_overframe")
    # plt.show()
    plt.close()

    plt.plot(IoUFrames[:,1])
    plt.ylabel("FN")
    plt.xlabel("#frame")
    plt.savefig(dir+"/FN_overframe")
    # plt.show()
    plt.close()

    plt.plot(IoUFrames[:,2])
    plt.ylabel("FP")
    plt.xlabel("#frame")
    plt.savefig(dir+"/FP_overframe")
    # plt.show()
    plt.close()

    plt.plot(IoUFrames[:,3])
    plt.ylabel("IoU")
    plt.xlabel("#frame")
    plt.savefig(dir+"/IoU_overframe")
    # plt.show()
    plt.close()


def plotF1(F1_frames, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    F1_frames = np.array(F1_frames)
    plt.plot(F1_frames[:,0])
    plt.ylabel("F1 score")
    plt.xlabel("#frame")
    plt.savefig(dir+"/F1_overframe")
    # plt.show()
    plt.close()

    plt.plot(F1_frames[:,1])
    plt.ylabel("precision")
    plt.xlabel("#frame")
    plt.savefig(dir+"/precision_overframe")
    # plt.show()
    plt.close()

    plt.plot(F1_frames[:,2])
    plt.ylabel("recall")
    plt.xlabel("#frame")
    plt.savefig(dir+"/recall_overframe")
    # plt.show()
    plt.close()
