import glob
import cv2
import numpy as np
from PIL import Image
import init_auv as auv
im_filename = '*.png'
imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = auv.imagesFilePath(imdir)
images = []

def nothing(x):
    pass

cv2.namedWindow('image')
# Creating trackbar:
cv2.createTrackbar('Threshold 1','image',0,1000,nothing)
cv2.createTrackbar('Threshold 2','image',0,1000,nothing)

num = 1
while(1):
    if auv.image_broken(filenames[num]):
        continue
    im = cv2.imread(filenames[num])
    thres1 = cv2.getTrackbarPos('Threshold 1','image')
    thres2 = cv2.getTrackbarPos('Threshold 2', 'image')
    gray = auv.preprocess_image(im)
    unsharp_im = cv2.Canny(gray,thres1,thres2)
    im_combined = np.hstack((gray,unsharp_im))
    cv2.imshow("image",im_combined)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('n'):
        num +=1

cv2.destroyAllWindows()