import math
import os

import cv2
import numpy as np
import matplotlib
import scipy
from scipy.ndimage.filters import gaussian_filter


def get_pose_img(keypoints):
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                [1,16], [16,18], [3,17], [6,18]]
    # visualize 2
    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    canvas = np.zeros((256,192,3),dtype = np.uint8) # B,G,R order

    for i in range(18):
        if keypoints[i][2]>0.01:
            canvas = cv2.circle(canvas, (int(keypoints[i][0]),int(keypoints[i][1])), 4, colors[i], thickness=-1)

    for i in range(17):
        index = np.array(limbSeq[i])-1
        if keypoints[index[1]][2]<0.01 or keypoints[index[0]][2]<0.01:
            continue
        cur_canvas = canvas.copy()
        Y = [keypoints[index[0]][0],keypoints[index[1]][0]]
        X = [keypoints[index[0]][1],keypoints[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas
 

def visualize_pose(pose_dict):
    keypoints = np.array(pose_dict["people"][0]["pose_keypoints"])
    keypoints = keypoints.reshape((-1, 3))
    pose_img = get_pose_img(keypoints)

    return pose_img


     
