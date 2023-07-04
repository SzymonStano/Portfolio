import numpy as np
import matplotlib.pyplot as plt


time = np.arange(1, 10**4)

differences = np.sqrt(np.diff(time))

for i in range(10):
    ys = np.zeros(len(time))
    for i in range(len(time)-1):
        ys[i+1] = np.random.normal(0, (differences[i]))

    ys = np.cumsum(ys)
    plt.scatter(time, ys, s=3)
iter_log = np.sqrt(2*time * np.log(np.log(time)))
plt.plot(time, iter_log)
plt.plot(time, -iter_log)
plt.show()
