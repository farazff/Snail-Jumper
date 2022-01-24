import matplotlib.pyplot as plt
import numpy as np

f = open("curve.txt", "r")
maxL = avgL = minL = []

for i in f.readlines():
    maxL.append(float(i.split(",")[0]))
    avgL.append(float(i.split(",")[1]))
    minL.append(float(i.split(",")[2]))

plt.plot(np.arange(0, len(maxL)), np.array(maxL), color='g', label="MAX")
plt.plot(np.arange(0, len(avgL)), np.array(avgL), color='r', label="AVG")
plt.plot(np.arange(0, len(minL)), np.array(minL), color='b', label="MIN")

plt.legend()
plt.show()
