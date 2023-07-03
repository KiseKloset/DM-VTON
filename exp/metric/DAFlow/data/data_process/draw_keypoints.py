import tqdm
import json
import numpy as np
import scipy
import math
import cv2
import argparse
from scipy.ndimage.filters import gaussian_filter
import glob
import matplotlib
import pylab as plt
import os
def get_poseimg(save_path,all_peaks,img_path = None):
	# find connection in the specified sequence, center 29 is in the position 15
	limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
	           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
	           [1,16], [16,18], [3,17], [6,18]]
	# the middle joints heatmap correpondence
	mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
	          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
	          [55,56], [37,38], [45,46]]
	# visualize 2
	stickwidth = 4

	colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
	          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
	          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
	cmap = matplotlib.cm.get_cmap('hsv')
	if img_path is None:
		canvas = np.zeros((256,192,3),dtype = np.uint8) # B,G,R order
	else:
		canvas = cv2.imread(img_path)

	for i in range(18):
	    rgba = np.array(cmap(1 - i/18. - 1./36))
	    rgba[0:3] *= 255
	    if all_peaks[i][2]>0.01:
	        cv2.circle(canvas, (int(all_peaks[i][0]),int(all_peaks[i][1])), 4, colors[i], thickness=-1)

	for i in range(17):
	    index = np.array(limbSeq[i])-1
	    if all_peaks[index[1]][2]<0.01 or all_peaks[index[0]][2]<0.01:
	        continue
	    cur_canvas = canvas.copy()
	    Y = [all_peaks[index[0]][0],all_peaks[index[1]][0]]
	    X = [all_peaks[index[0]][1],all_peaks[index[1]][1]]
	    mX = np.mean(X)
	    mY = np.mean(Y)
	    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
	    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
	    polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
	    cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
	    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
	cv2.imwrite(save_path,canvas)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)

    parser.add_argument('--save_path', type=str, default="./out")
    opt = parser.parse_args()
    save_path = opt.save_path
    if not os.path.exists(save_path):
    	os.mkdir(save_path)
    if opt.json_path[:-3] == "json":
    	all_files = [opt.json_path]
    else:
        print(os.path.join(opt.json_path,"*json"))
        all_files = glob.glob(os.path.join(opt.json_path,"*json"))
    for file in tqdm.tqdm(all_files):
    	if "cloth" in file:
    	  continue
    	keypoints = json.load(open(file))["people"][0]["pose_keypoints"]
    	keypoints = np.resize(keypoints,(18,3))
    	save_path_ = os.path.join(save_path,file.split('/')[-1].replace('json','jpg'))
    	get_poseimg(save_path_,keypoints)
if __name__ == "__main__":
	main()
