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
timestamp = []
select = 2
filepath = ''.join((imdir,'/',im_filename))
blob_params = dict(minThreshold = 10,
                   maxThreshold = 200,
                   filterByArea=True,
                   minArea=1500,
                   filterByCircularity=True,
                   minCircularity=0.1,
                   filterByConvexity=True,
                   minConvexity=0.87,
                   filterByInertia=True,
                   minInertiaRatio=0.01)
#im = cv2.imread('/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1/20181005_084743_424.png',0)
for imfile in glob.glob(filepath):
    filenames.append(imfile)


filenames.sort()

# Set up the detector with default parameters.

# https://www.learnopencv.com/blob-detection-using-opencv-python-c/
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)

for i,filename in enumerate(filenames):
    if image_broken(filename):
        continue

    print i

    im = cv2.imread(filename)
    im = cv2.resize(im,(640,480))
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if select == 1:
        res_name = "Clahe"
        res = clahe.apply(im_gray) # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    elif select == 2:
        res_name = "Histogram Equalization"
        res = cv2.equalizeHist(im_gray)
    #im_combined = np.hstack((im,cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)))

    # Detect blobs.
    keypoints = detector.detect(im_gray)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_gray_with_keypoints = cv2.drawKeypoints(im_gray, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_gray_with_keypoints = cv2.putText(im_gray_with_keypoints,"Keypoints: " + str(np.size(keypoints)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    keypoints = detector.detect(res)
    res_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    res_with_keypoints = cv2.putText(res_with_keypoints, "Keypoints: " + str(np.size(keypoints)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2)
    im_combined = np.hstack((im_gray_with_keypoints, res_with_keypoints))
    cv2.imshow('Recordings: Mission 1',im_combined)
    if cv2.waitKey(100) & 0xFF  == ord('q'): #Pause
        key = cv2.waitKey()
        if cv2.waitKey() & 0xFF  == ord('q'): #Exiting
            break
        elif cv2.waitKey() & 0xFF  == ord('1'): #Exiting
            select = 1
        elif cv2.waitKey() & 0xFF  == ord('2'): #Using
            select = 2

print('Ending program')
cv2.destroyAllWindows()

