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

def draw_flow(img,p1,p2,mask=None):
    if mask is None:
        p1 = np.reshape(p1,(-1 , 2))
        p2 = np.reshape(p2,(-1 , 2))
        if len(p1) > 0:
            for i, (new,old) in enumerate(zip(p1,p2)):
                a,b = new.ravel()
                c,d = old.ravel()
                img = cv2.line(img,(a,b),(c,d),(0,255,0),1)
    else:
        p1_inliers = p1[mask==1]
        p2_inliers = p2[mask == 1]
        p1_outliers = p1[mask == 0]
        p2_outliers = p2[mask == 0]
        if len(p1_inliers) > 0:
            p1_inliers = np.reshape(p1_inliers,(-1,2))
            p2_inliers = np.reshape(p2_inliers, (-1, 2))
            for i, (new, old) in enumerate(zip(p1_inliers, p2_inliers)):
                a, b = new.ravel()
                c, d = old.ravel()
                img = cv2.line(img, (a, b), (c, d), (0, 255, 0), 1)
        if len(p1_outliers) > 0:
            p1_outliers = np.reshape(p1_outliers, (-1, 2))
            p2_outliers = np.reshape(p2_outliers, (-1, 2))
            for i, (new, old) in enumerate(zip(p1_outliers, p2_outliers)):
                a, b = new.ravel()
                c, d = old.ravel()
                img = cv2.line(img, (a, b), (c, d), (0, 0, 255), 1.5)
    return img

def sparse_optical_flow(img1,img2,points,fb_threshold,optical_flow_params):
    #old_points = points.copy()
    new_points, status , err = cv2.calcOpticalFlowPyrLK(img1,img2,points,None,**optical_flow_params)
    if fb_threshold>0:
        new_points_r, status_r, err = cv2.calcOpticalFlowPyrLK(img2,img1,new_points,None,**optical_flow_params)
        new_points_r[status_r==0] = False#np.nan
        fb_good = (np.fabs(new_points_r-points) < fb_threshold).all(axis=2)
        new_points[~fb_good] = np.nan
        old_points = np.reshape(points[~np.isnan(new_points)],(-1,1,2))
        new_points = np.reshape(new_points[~np.isnan(new_points)],(-1,1,2))
    return new_points,old_points

def update_motion(points1,points2,Rpos,tpos,cam_mat=None,scale = 1.0):
    E, mask = cv2.findEssentialMat(points1,points2,cameraMatrix=cam_mat,method=cv2.LMEDS,prob=0.999, mask=None)
    newmask = np.copy(mask)
    _,R,t,newmask = cv2.recoverPose(E,points1,points2,cameraMatrix=cam_mat,mask=newmask)
    tp = tpos+np.dot(Rpos,t)*scale
    Rp = np.dot(R,Rpos)
    return Rp,tp,mask

def find_features(im,method=0):
    points = []
    if method == 0:
        keypoints = blob.detect(im)
    elif method == 1:
        keypoints = fast.detect(im)

    for point in keypoints:
        points.append(point.pt)
    points = np.array(points, dtype=np.float32)

    return np.reshape(points, (-1, 1, 2)),keypoints

def preprocess_image(im,size=(640,480),method=0):
    im = cv2.resize(im, size)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if method==0:
        res = cv2.equalizeHist(gray)
    elif method == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        res = clahe.apply(gray)
    return res

def draw_path(canvas,points,scale=0.1,clear=True,color=(0,255,0)):
    points = np.reshape(points,(-1,3))
    dim = np.shape(canvas)
    canvas_h = dim[0]
    canvas_w = dim[1]
    canvas = np.zeros(dim,dtype=np.uint8)

    pos_orig = points[0]
    x_orig = pos_orig[0]
    y_orig = pos_orig[1]

    for i, p2 in enumerate(points):
        p1 = points[i-1]
        x1 = int((p1[0]-x_orig)*scale+canvas_w/2)
        y1 = int((p1[1] - y_orig) * scale + canvas_h / 2)
        x2 = int((p2[0] - x_orig) * scale + canvas_w / 2)
        y2 = int((p2[1] - y_orig) * scale + canvas_h / 2)
        cv2.line(canvas,(x1,y1),(x2,y2),color,1)
    return canvas

cam_mat = np.array([[1417.84363, 0.0, 353.360213], [0.0, 1469.27135, 270.122002],  [0.0, 0.0, 1.0]], dtype=np.float32)
im_filename = '*.png'
imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
#imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
#imdir = 'C:/Users/Rasmus/Dropbox/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = []
images = []
path = []
timestamp = []
backward_flow_threshold = 3
prev_frame = None
canvas = np.zeros((640, 480, 3), dtype=np.uint8)
select = 2
filepath = ''.join((imdir,'/',im_filename))
fast_params = dict(threshold = 30,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

optical_flow_params = dict(winSize = (10,10),
                           maxLevel = 4,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
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
params.maxThreshold = 300;

# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
prev_points = []
feat_method = 0
Rpos = np.eye(3,3,dtype=np.float32)
tpos = np.zeros((3,1),dtype=np.float32)
blob = cv2.SimpleBlobDetector_create(params)
fast = cv2.FastFeatureDetector_create(**fast_params)
for i,frame in enumerate(filenames):
    if image_broken(frame):
        continue

    gray = preprocess_image(cv2.imread(frame),(640,480),0)
    im = cv2.resize(cv2.imread(frame),(640,480))
    if prev_frame is None:
        prev_points,keypoints = find_features(gray,0)
        prev_frame = gray.copy()
        continue

    if len(prev_points) < 100000:
        prev_points,keypoints = find_features(prev_frame, feat_method)
    print "Number of features: " + str(len(prev_points))
    new_points, prev_points = sparse_optical_flow(prev_frame, gray, prev_points, backward_flow_threshold,
                                                  optical_flow_params)
    path.append(tpos)
    old_tpos = np.copy(tpos)
    #Rpos, tpos, mask = update_motion(prev_points, new_points, Rpos, old_tpos, cam_mat)
    flow_im = draw_flow(im,prev_points,new_points)
    #canvas = draw_path(canvas,path)
    gray_with_kp = cv2.drawKeypoints(gray,keypoints,np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_combined = np.hstack((flow_im, gray_with_kp))
    cv2.imshow('Recordings: Mission 1',im_combined)
    prev_frame = gray.copy()
    prev_points = new_points.copy()
    if cv2.waitKey(100) & 0xFF  == ord('q'): #Pause
        key = cv2.waitKey()
        if cv2.waitKey() & 0xFF  == ord('q'): #Exiting
            break
        elif cv2.waitKey() & 0xFF  == ord('1'):
            feat_method = 1
        elif cv2.waitKey() & 0xFF  == ord('2'): #Using
            feat_method = 0

print('Ending program')
cv2.destroyAllWindows()

