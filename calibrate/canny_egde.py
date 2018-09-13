import glob
import cv2
import numpy as np
from PIL import Image

def image_broken(img_path):
    try:
        im = Image.open(img_path)
        im.verify()
        im.close()
        return 0
    except:
        return 1
im_filename = '*.png'
imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = []
images = []
filepath = ''.join((imdir,'/',im_filename))

for imfile in glob.glob(filepath):
    filenames.append(imfile)
filenames.sort()


def nothing(x):
    pass

cv2.namedWindow('image')
# Creating trackbar:
cv2.createTrackbar('Threshold 1','image',0,1000,nothing)
cv2.createTrackbar('Threshold 2','image',0,1000,nothing)

num = 1
while(1):
    im = cv2.imread(filenames[num], 0)
    thres1 = cv2.getTrackbarPos('Threshold 1','image')
    thres2 = cv2.getTrackbarPos('Threshold 2', 'image')
    gaussian_im = cv2.GaussianBlur(im, (9, 9), 10.0)
    # unsharp_im = cv2.addWeighted(im,alpha,gaussian_im,-beta,gamma)
    unsharp_im = cv2.addWeighted(im, 0.9, gaussian_im, -0.5, 0)
    unsharp_im = cv2.equalizeHist(unsharp_im)
    im = cv2.equalizeHist(im)
    unsharp_im = cv2.Canny(unsharp_im,thres1,thres2)
    im_combined = np.hstack((im,unsharp_im))
    cv2.imshow("image",im_combined)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('n'):
        num +=1

cv2.destroyAllWindows()