import numpy as np
import matplotlib.pyplot as plt


time = np.linspace(0, 2000, 2000)

differences = np.sqrt(np.diff(time))

trajectories = 1000

ys = np.zeros((len(time), trajectories))
variance = np.empty((len(time)-1))
for i in range(len(time)-1):
    ys[i+1] = np.random.normal(0, (differences[i]), size=trajectories)


ys = np.cumsum(ys, axis=0)    # wertykalnie
variance = np.var(ys, axis=1)   # horyzontalnie
plt.scatter(time, variance)
plt.show()


