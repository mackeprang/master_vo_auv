import h5py
import matplotlib.pyplot as plt

filename = "/Users/Mackeprang/Dropbox (Personlig)/Master Thesis/Data/20180910 Optical flowtest/20181010_105618_Mission_1/output.h5"

f = h5py.File(filename,'r')

print list(f["Position"])
alt = f["Position"]["Altitude"]
depth = f["Position"]["Depth"]
acc_x = f["Position"]["Acc_X"]
acc_y = f["Position"]["Acc_Y"]
acc_z = f["Position"]["Acc_Z"]

plt.plot(acc_x)
plt.plot(acc_y)
plt.plot(acc_z)
ax = plt.gca()
plt.ylabel("Altitude")
ax.grid()
plt.show()

# plt.close("all")