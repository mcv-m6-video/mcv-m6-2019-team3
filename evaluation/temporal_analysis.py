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

def plotIoU_by_frame(IoUFrames, dir):
    IoUFrames = np.array(IoUFrames)

    #plotIoU_by_frame_structure(IoUFrames[:,0], dir, "/TP_overframe", "TP")
    plot_by_frame_structure(IoUFrames[:,1], dir, "/FN_overframe", "FN", max(IoUFrames[:,1]))
    plot_by_frame_structure(IoUFrames[:,2], dir, "/FP_overframe", "FP", max(IoUFrames[:,2]))
    plot_by_frame_structure(IoUFrames[:,3], dir, "/IoU_overframe", "IoU", max(IoUFrames[:,3]))

def plotF1_by_frame(F1_frames, dir):
    F1_frames = np.array(F1_frames)

    plot_by_frame_structure(F1_frames[:,0], dir, "/F1_overframe", "F1 Score", max(F1_frames[:,0]))
    plot_by_frame_structure(F1_frames[:,1], dir, "/precision_overframe", "Precision", max(F1_frames[:,1]))
    plot_by_frame_structure(F1_frames[:,2], dir, "/recall_overframe", "Recall", max(F1_frames[:,2]))

def plot_by_frame_structure(IoUFrames, dir, folder, ylabel, max_y):
    if not os.path.exists(dir+folder):
        os.makedirs(dir+folder)
    IoUFrames = np.array(IoUFrames)

    for n, IoUFrame in enumerate(IoUFrames):
        if n == 0:
            plt.plot(IoUFrames[0])
        else:
            plt.plot(IoUFrames[0:n])

        plt.ylabel(ylabel)
        plt.xlabel("#frame")
        plt.axis([0, len(IoUFrames), -0.1, max_y + 0.1])
        plt.savefig(dir+folder+"/{:04d}.png".format(n))
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
