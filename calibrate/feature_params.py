import cv2
def get_FAST_params():
    return dict(threshold = 30,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

def get_optical_flow_params():

    return dict(winSize = (15,15),
                           maxLevel = 3,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
def get_good_feature_params():
    return dict(maxCorners=20000,
                               qualityLevel=0.01,
                               minDistance=5,
                               blockSize=5,
                               useHarrisDetector=False)

def get_blob_params(): # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
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
    return params