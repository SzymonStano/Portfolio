import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

time = []
lmbd = 1
sum = 0
T = 1000
while sum < T:
    t = np.random.exponential(1/lmbd)
    sum += t
    time.append(sum)

time2 = [0] + time  # zero potrzebne do rysowania poziomych linii
ys = np.random.normal(size=len(time))
ys = np.cumsum(ys)
plt.scatter(time, ys)
ys2 = np.insert(ys, 0, 0)
for i in range(len(time)):
    plt.hlines(ys2[i], time2[i], time[i])
plt.show()


time = []
lmbd = 1
sum = 0
# T = 1000
while sum < T:
    t = np.random.exponential(1/lmbd)
    sum += t
    time.append(sum)

time2 = [0] + time  # zero potrzebne do rysowania poziomych linii
ys = np.random.standard_cauchy(size=len(time))
ys = np.cumsum(ys)
# ys = np.arange(1, len(time)+1)
ys2 = np.insert(ys, 0, 0)
plt.scatter(time, ys)
# plt.scatter(time, ys-1, facecolors='none', edgecolors='blue', alpha=0.5)
for i in range(len(time)):
    plt.hlines(ys2[i], time2[i], time[i])
plt.show()


