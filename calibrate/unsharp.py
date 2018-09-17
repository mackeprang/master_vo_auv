import glob
import cv2
import numpy as np
from PIL import Image
import init_auv as auv
import feature_params as feat_params
imdir = 'Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = auv.imagesFilePath(imdir)

def nothing(x):
    pass

cv2.namedWindow('image')
# Creating trackbar:
cv2.createTrackbar('Alpha','image',0,255,nothing)
cv2.createTrackbar('Beta','image',0,255,nothing)
cv2.createTrackbar('Gamma','image',0,255,nothing)

num = 1
while(1):
    im = cv2.imread(filenames[num], 0)
    alpha = cv2.getTrackbarPos('Alpha','image')/128.0
    beta = cv2.getTrackbarPos('Beta','image')/128.0
    gamma = cv2.getTrackbarPos('Gamma', 'image')
    gaussian_im = cv2.GaussianBlur(im, (9, 9), 10.0)
    unsharp_im = cv2.addWeighted(im,alpha,gaussian_im,-beta,gamma)
    # unsharp_im = cv2.addWeighted(im, 0.9, gaussian_im, -0.5, 0)
    unsharp_im = cv2.equalizeHist(unsharp_im)
    im = cv2.equalizeHist(im)
    im_combined = np.hstack((im,unsharp_im))
    cv2.imshow("image",im_combined)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        print"Alpha: " + str(alpha)
        print"Beta: " + str(-beta)
        print"Gamma: " + str(gamma)
        break
    if key & 0xFF == ord('n'):
        num += 1

cv2.destroyAllWindows()