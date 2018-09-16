import glob
import cv2
import numpy as np
from PIL import Image
import  init_auv as auv
im_filename = '*.png'
imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = auv.imagesFilePath(imdir)
images = []
prev_frame = None

hsv = np.zeros_like(cv2.imread(filenames[0]))
hsv[...,1] = 255
for i,frame in enumerate(filenames):
    if auv.image_broken(frame):
        continue
    img = cv2.imread(frame)
    gray = auv.preprocess_image(img,size=None)

    if prev_frame is None:
        prev_frame = gray
        continue
    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 30, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Match", bgr)
    key = cv2.waitKey()
    if key & 0xFF == ord('q'):
        break
    prev_frame = gray.copy()
cv2.destroyAllWindows()