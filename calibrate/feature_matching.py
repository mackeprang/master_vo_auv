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

cam_mat = np.loadtxt("cam_mat.csv", dtype=np.float32, delimiter=',')
dist_coeff = np.loadtxt("dist_coeff.csv", dtype=np.float32, delimiter=',')
im_filename = '*.png'
imdir = '/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Pictures/20181005_084733.9640_Mission_1'
filenames = []
images = []
filepath = ''.join((imdir,'/',im_filename))

for imfile in glob.glob(filepath):
    filenames.append(imfile)

filenames.sort()

prev_frame = None
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
for i,frame in enumerate(filenames):
    if image_broken(frame):
        continue
    img = cv2.imread(frame)
    img = cv2.resize(img,(640,480))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if prev_frame is None:
        prev_frame = gray
        kp1,des1 = orb.detectAndCompute(prev_frame,None)
    print "Numbers of features: " + str(len(kp1))
    kp2, des2 = orb.detectAndCompute(gray, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_match = cv2.drawMatches(prev_frame, kp1, gray, kp2, matches[:20],None, flags=2)
    cv2.imshow("Match",img_match)
    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        break
    kp1 = kp2
    des1 = des2
    prev_frame = gray.copy()
cv2.destroyAllWindows()