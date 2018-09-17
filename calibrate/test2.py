import glob
import cv2
import numpy as np
from PIL import Image
import init_auv as auv
import feature_params as feat_params

def find_features(im,method=0):
    points = []
    keypoints = []
    if method == 0:
        keypoints = blob.detect(im)
    elif method == 1:
        keypoints = fast.detect(im)
    elif method == 2:
        points = cv2.goodFeaturesToTrack(im, mask=None, **feat_params.get_good_feature_params())
        for kp in points:
            kp =  kp[0]
            keypoints.append(cv2.KeyPoint(kp[0],kp[1],_size=2))

        return points,keypoints
    for point in keypoints:
        points.append(point.pt)
    points = np.array(points, dtype=np.float32)

    return np.reshape(points, (-1, 1, 2)),keypoints

cam_mat = auv.getCamMat()
dist_coeff = auv.getCamMat()
imdir = 'Master Thesis/Pictures/20181005_084733.9640_Mission_1'
#imdir = 'Master Thesis/Pictures/20181010_111618.6170_Mission_4'
filenames = auv.imagesFilePath(imdir)
#filenames = filenames[1000:]
images = []
path = []
timestamp = []
backward_flow_threshold = 1
prev_frame = None
canvas = np.zeros((480,300,3), dtype=np.uint8)
mask = None

# Set up the detector with default parameters.


prev_points = []
preproc_method = 1
feat_method = 2 # 0: Blob, 1: FAST, 2: GoodFeatures to Track
Rpos = np.eye(3,3,dtype=np.float32)
tpos = np.zeros((3,1),dtype=np.float32)
blob = cv2.SimpleBlobDetector_create(feat_params.get_blob_params())
fast = cv2.FastFeatureDetector_create(**feat_params.get_FAST_params())
for i,frame in enumerate(filenames):
    if auv.image_broken(frame):
        continue
    im = cv2.imread(frame)
    gray = auv.preprocess_image(im,method=preproc_method)
    im = cv2.resize(im,auv.IM_PRE_RESOLUTION)

    if prev_frame is None:
        prev_points,keypoints = find_features(gray,feat_method)
        prev_frame = gray.copy()
        continue

    if len(prev_points) < 100000:
        prev_points,keypoints = find_features(prev_frame, feat_method)
    print "Number of features: " + str(len(prev_points))
    new_points, prev_points = auv.sparse_optical_flow(prev_frame, gray, prev_points, backward_flow_threshold,
                                                  feat_params.get_optical_flow_params())

    path.append(tpos)
    old_tpos = np.copy(tpos)
    Rpos, tpos, mask = auv.update_motion(prev_points, new_points, Rpos, old_tpos, cam_mat)
    flow_im = auv.draw_flow(im, prev_points, new_points, mask=mask)
    canvas = auv.draw_path(canvas,path,scale=2)
    gray_with_kp = cv2.drawKeypoints(gray,keypoints,np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_combined = np.hstack((flow_im, canvas))
    im_combined = np.hstack((im_combined, gray_with_kp))
    cv2.imshow('Recordings: Mission 1',im_combined)
    # cv2.imshow('Canvas', canvas)
    prev_frame = gray.copy()
    prev_points = new_points.copy()
    key = cv2.waitKey(100)
    if key & 0xFF  == ord('q'): #Pause
        key = cv2.waitKey()
        if key & 0xFF  == ord('q'): #Exiting
            break
        elif key & 0xFF  == ord('1'):
            feat_method = 1
        elif key & 0xFF  == ord('2'): #Using
            feat_method = 0
        elif key & 0xFF  == ord('3'): #Using
            feat_method = 2
print('Ending program')
cv2.destroyAllWindows()

