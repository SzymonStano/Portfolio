import numpy as np
import matplotlib.pyplot as plt


time = np.linspace(0, 2000, 2000)


ys = np.zeros(len(time))
for i in range(1, len(time)):
    ys[i] = np.random.normal(0, (time[i] - time[i-1])**2)

ys = np.cumsum(ys)
plt.step(time, ys)
plt.show()


ys = np.zeros(len(time))
zs = np.zeros(len(time))
for i in range(1, len(time)):
    ys[i] = np.random.normal(0, np.sqrt((time[i] - time[i-1])))
    zs[i] = np.random.normal(0, np.sqrt((time[i] - time[i-1])))

ys = np.cumsum(ys)
zs = np.cumsum(zs)

plt.plot(ys, zs)
plt.show()


xs = np.zeros(len(time))
ys = np.zeros(len(time))
zs = np.zeros(len(time))
for i in range(1, len(time)):
    ys[i] = np.random.normal(0, (time[i] - time[i-1])**2)
    xs[i] = np.random.normal(0, (time[i] - time[i-1])**2)
    zs[i] = np.random.normal(0, (time[i] - time[i-1])**2)

xs = np.cumsum(xs)
ys = np.cumsum(ys)
zs = np.cumsum(zs)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xs, ys, zs)
plt.show()
