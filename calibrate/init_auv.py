import numpy as np
import cv2
from PIL import Image
import h5py
import utm
import glob
from sys import platform
# CONSTANTS
IM_PRE_EQ_HIST = 0
IM_PRE_CLASE = 1
IM_PRE_RESOLUTION = (640,480)
SYS_PLATFORM = platform
################ GETTING DATA ################

def get_sys_platform(platform=SYS_PLATFORM):
    if platform == "darwin":
        return 0
    elif platform == "win32":
        return 1

def image_broken(img_path):
    try:
        im = Image.open(img_path)
        im.verify()
        im.close()
        return 0
    except:
        return 1

def read_h5(file_path):
    return h5py.File(file_path, 'r')

def getAccData(hdf):
    f = read_h5(hdf)
    acc_x = f["Position"]["Acc_X"]
    acc_y = f["Position"]["Acc_Y"]
    acc_z = f["Position"]["Acc_Z"]

    return {"Acc_X": acc_x,"Acc_Y":acc_y,"Acc_Z":acc_z}

def getRelPos(hdf):
    f = read_h5(hdf)
    lat = f["Position"]["Lat"]
    lon = f["Position"]["Lon"]
    xy = []
    for x, y in zip(lat, lon):
        xy.append(utm.from_latlon(x, y))

    rel_pos_x = []
    rel_pos_y = []
    for pos in xy:
        rel_pos_x.append(pos[0] - xy[0][0])
        rel_pos_y.append(pos[1] - xy[0][1])

    return {"X":rel_pos_x,"Y":rel_pos_y}

def getCamMat():
    return np.loadtxt("cam_mat.csv", dtype=np.float32, delimiter=',')

def getDistMat():
    return np.loadtxt("dist_coeff.csv", dtype=np.float32, delimiter=',')

def imagesFilePath(imdir,ext='*.png'):
    if get_sys_platform() == 0:
        start_path = '/Users/Mackeprang/Dropbox (Personlig)/'
    elif get_sys_platform() == 1:
        start_path = 'C:/Users/Rasmus/Dropbox/'

    filenames = []
    filepath = ''.join((start_path,imdir, '/', ext))

    for imfile in glob.glob(filepath):
        filenames.append(imfile)
    filenames.sort()
    return filenames

def datasetPath_Mads(num=0):
    if get_sys_platform() == 0:
        start_path = '/Users/Mackeprang/Dropbox (Personlig)/'
    elif get_sys_platform() == 1:
        start_path = 'C:/Users/Rasmus/Dropbox/'
    if num == 0:
        images = ''.join((start_path,'Master Thesis/Pictures/20181005_084733.9640_Mission_1'))
        data = ''.join((start_path,'Master Thesis/Data/20180910 Optical flowtest/20181010_122400_Mission_5/output.h5'))
    if num == 1:
        pass
    return {"Images": images,"Data": data}

################# OpenCV Functions ##################
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

def preprocess_image(im_color,size=IM_PRE_RESOLUTION,method=IM_PRE_EQ_HIST, unsharp=True):
    if size != None:
        im_color = cv2.resize(im_color, size)
    gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
    if unsharp:
        gaussian_im = cv2.GaussianBlur(gray, (9, 9), 10.0)
        gray = cv2.addWeighted(gray, 0.9, gaussian_im, -0.5, 0)

    if method==IM_PRE_EQ_HIST:
        res = cv2.equalizeHist(gray)
    elif method == IM_PRE_CLASE:
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