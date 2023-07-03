from predict_pose import PosePredictor
from visualize import visualize_pose
import cv2

if __name__ == "__main__":
    p = PosePredictor(device_id=2)
    img = cv2.imread('nnkn2.jpg')
    k = p.generate_pose_keypoints(img)
    pose_img = visualize_pose(k)
    cv2.imwrite("out.jpg", pose_img)