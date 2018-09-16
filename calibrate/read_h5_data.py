import init_auv as auv
import matplotlib.pyplot as plt
import utm
import numpy as np
filename = "/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Data/20180910 Optical flowtest/20181010_122400_Mission_5/output.h5"


f = auv.read_h5(filename)
acc_data = auv.getAccData(filename)

#print list(f["Position"])
#print acc_data["Acc_X"].value
alt = f["Position"]["Altitude"]
depth = f["Position"]["Depth"]
acc_x = acc_data["Acc_x"]
acc_y = f["Position"]["Acc_Y"]
acc_z = f["Position"]["Acc_Z"]
lat = f["Position"]["Lat"]
lon = f["Position"]["Lon"]
xy = []
for x,y in zip(lat,lon):
    xy.append(utm.from_latlon(x,y))

rel_pos_x = []
rel_pos_y = []
for pos in xy:
    rel_pos_x.append(pos[0]-xy[0][0])
    rel_pos_y.append(pos[1] - xy[0][1])
plt.close("all")
fig1 = plt.figure()
thr = np.ones_like(alt)
plt.scatter(rel_pos_x,rel_pos_y)
fig2 = plt.figure()
thr = np.ones_like(alt)
thr *= 1.6
plt.plot(alt)
plt.plot(thr)
fig3 = plt.figure()
plt.plot(acc_y)
plt.plot(acc_x)
plt.plot(acc_z)
ax = plt.gca()
ax.legend(["Acc Y","Acc X", "Acc Z"])
plt.ylabel("Altitude")
ax.grid()
plt.show()

plt.close("all")