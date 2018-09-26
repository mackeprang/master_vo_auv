import glob
import cv2
import numpy as np
from PIL import Image
import init_auv as auv
import os
import math
import time
import feature_params as feat_params
import matplotlib.pyplot as plt

save_flag = False

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
mission = auv.get_mission_by_num(11)
f = auv.read_h5(mission["Data"])
heading = f["Position"]["Heading"]
alt = auv.get_altitude(mission)
save_data = open('vo_data.txt','w')
cam_mat = auv.getCamMat()
dist_coeff = auv.getCamMat()
start_idx = 266
filenames = auv.find_images(mission["Images"])
print len(filenames)
gps_pos_x,gps_pos_y,goal_idx = auv.get_gps_goal(auv.getRelPos(mission["Data"])["X"],auv.getRelPos(mission["Data"])["Y"])
filenames = filenames[start_idx:goal_idx]
print "After " + str(len(filenames))

heading = heading[start_idx:goal_idx]
alt = alt[start_idx:goal_idx]
heading_rad = (heading[0]*np.pi)/180

images = []
path = []
timestamp = []
backward_flow_threshold = 1
prev_frame = None
canvas = np.zeros((480,400,3), dtype=np.uint8)
mask = None
gps_pos_x = int(gps_pos_x + 400//2)
gps_pos_y = int(gps_pos_y + 480//2)

# Set up the detector with default parameters.

alpha = 0.5
prev_points = []
preproc_method = 1
feat_method = 2 # 0: Blob, 1: FAST, 2: GoodFeatures to Track
Rpos = np.array([[np.cos(heading_rad),-np.sin(heading_rad),0],
                 [np.sin(heading_rad),np.cos(heading_rad),0],
                 [0,0,1]])
#Rpos = np.array([[1,0,0],[0,np.cos(heading_rad),-np.sin(heading_rad)],[0,np.sin(heading_rad),np.cos(heading_rad)]])#np.eye(3,3,dtype=np.float32)
#print Rpos
#print np.eye(3,3,dtype=np.float32)
gps_x = auv.getRelPos(mission["Data"])["X"]
gps_y = auv.getRelPos(mission["Data"])["Y"]
dtpos = None
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
    #print "Number of features: " + str(len(prev_points))
    new_points, prev_points = auv.sparse_optical_flow(prev_frame, gray, prev_points, backward_flow_threshold,
                                                  feat_params.get_optical_flow_params())
    if i > 286:
        pass
        #cv2.waitKey()

    path.append(tpos)
    old_tpos = np.copy(tpos)
    old_Rpos = np.copy(Rpos)

    Rpos, tpos, mask = auv.update_motion(prev_points, new_points, Rpos, old_tpos, cam_mat,scale=0.6)
    heading_rad = heading[i]*np.pi/180
    #Rpos = np.array([[np.cos(heading_rad), -np.sin(heading_rad), 0],
    #                 [np.sin(heading_rad), np.cos(heading_rad), 0],
    #                [0, 0, 1]])
    #Rpos = old_Rpos * (1 - alpha) + Rpos * alpha
    #tpos = old_tpos * (1 - alpha) + tpos * alpha
    tpos[2] = alt[i]

    flow_im = auv.draw_flow(im, prev_points, new_points, mask=mask)
    canvas = auv.draw_path(canvas,path,scale=0.3,flipX=False,rotate=1,flipY=False,with_gps_marks=True,gps_x=gps_x,gps_y=gps_y)
    canvas = cv2.circle(canvas,(gps_pos_x,gps_pos_y),5,(0,0,255))

    status_str = "Frame " + str(i) + "/"+str(len(filenames))
    canvas = cv2.putText(canvas,status_str,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0))
    canvas = cv2.putText(canvas, "Features: " + str(len(new_points)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    canvas = cv2.putText(canvas, "Measured Heading: " + str(auv.get_current_heading(Rpos)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    canvas = cv2.putText(canvas, "Compass Heading: " + str(heading[i]), (20, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    canvas = cv2.putText(canvas, "Altitude: " + str(alt[i]), (20,100),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    gray_with_kp = cv2.drawKeypoints(gray,keypoints,np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    canvas = cv2.putText(canvas, "Tpos: " + np.array_str(tpos,precision=2,suppress_small=True), (20, 120),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    im_combined = np.hstack((flow_im, canvas))
    im_combined = np.hstack((im_combined, gray_with_kp))
    cv2.imshow(mission["Name"],im_combined)
    # cv2.imshow('Canvas', canvas)
    prev_frame = gray.copy()
    prev_points = new_points.copy()
    if save_flag == True:
        x,y,z = tpos
        np.savetxt(save_data,tpos,fmt="%.5f",delimiter=',', newline=',')
        np.savetxt(save_data, Rpos, fmt="%.5f", delimiter=',',newline=',')
        save_data.write("\n")
        cv2.imwrite("mission_" + mission["Name"] + ".png", im_combined)
    key = cv2.waitKey(1)
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

save_data.close()
print('Ending program')
cv2.waitKey()
cv2.destroyAllWindows()

