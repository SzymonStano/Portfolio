import numpy as np
import matplotlib.pyplot as plt

"Procesy Levy'ego dla zadanych czasów"
"Proces gamma, variance gamma, symetryczny stabilny"

time = np.arange(1, 1000, 1)

differences = np.sqrt(np.diff(time))

trajectories = 5
ys = np.zeros((len(time), trajectories))
for i in range(len(time)-1):
    ys[i+1] = np.random.gamma((differences[i]), 1, size=trajectories)

ys = np.cumsum(ys, axis=0)
plt.plot(time, ys)
plt.title("Proces gamma")
plt.show()


time = np.arange(0, 1500, 0.1)
differences = np.sqrt(np.diff(time))
trajectories = 5
ys = np.zeros((len(time), trajectories))
for i in range(len(time)-1):
    sigmas = np.random.gamma((differences[i]), 1, trajectories)
    samples = np.random.normal(0, sigmas[:, np.newaxis],  (trajectories, 1))
    ys[i+1] = np.squeeze(samples)


ys = np.cumsum(ys, axis=0)
plt.plot(time, ys)
plt.title("Proces variance gamma")
plt.show()


dt = 0.1
alpha = 1.2
time = np.arange(0, 200, dt)
n = len(time)
trajectories = 5
for i in range(trajectories):
    U = np.random.uniform(-0.5, 0.5, n)
    W = np.random.exponential(1, n)
    ys = dt**(1/alpha)*np.sin(alpha*U)/(np.cos(U))**(1/alpha)*(np.cos((1-alpha)*U)/W)**((1-alpha)/alpha)
    ys = np.cumsum(ys)
    plt.plot(time, ys)
plt.title("Proces stabilny symetryczny")
plt.show()

