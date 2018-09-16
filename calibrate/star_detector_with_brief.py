import glob
import cv2
import numpy as np
from PIL import Image
import init_auv as auv
import feature_params as feat_params
cam_mat = auv.getCamMat()
dist_coeff = auv.getDistMat()
im_filename = '*.png'
imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = auv.imagesFilePath(imdir)
images = []

prev_frame = None

detector = cv2.SimpleBlobDetector_create(feat_params.get_blob_params())
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
for i,frame in enumerate(filenames):
    if auv.image_broken(frame):
        continue
    img = cv2.imread(frame)
    gray = auv.preprocess_image(img)

    if prev_frame is None:
        prev_frame = gray
        kp1 = detector.detect(gray, None)
        _,des1 = orb.compute(gray, kp1)
        continue
    print "Numbers of features: " + str(len(kp1))
    kp2 = detector.detect(gray,None)
    _,des2 = orb.compute(gray,kp2)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_match = cv2.drawMatches(prev_frame, kp1, gray, kp2, matches[:10], None, flags=2)
    cv2.imshow("Match", img_match)
    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        break
    kp1 = kp2
    des1 = des2
    prev_frame = gray.copy()
cv2.destroyAllWindows()