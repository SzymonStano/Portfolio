import numpy as np
import matplotlib.pyplot as plt
from numba import njit



u = 5   # kapitał_początkowy
c = 0.5     # dryf
intensywność = 1
lmbd = 1

time = []
sum_time = 0
y_value = u
plotter = []
T = 20
time.append(sum_time)
plotter.append(y_value)
while sum_time < T:
    t = np.random.exponential(1/intensywność)
    L = np.random.exponential(1/lmbd)
    sum_time += t
    time.append(sum_time)
    time.append(sum_time)
    y_value += c*t
    plotter.append(y_value)
    y_value -= L
    plotter.append(y_value)
    if y_value < 0:
        break


plt.plot(time, plotter)
plt.show()


@njit
def probability(Time, attempts):
    loses_counter = 0
    for i in range(attempts):
        sum_time = 0
        y_value = u
        while sum_time < Time:
            t = np.random.exponential(1/intensywność)
            sum_time += t
            if sum_time > Time:
                break
            L = np.random.exponential(1/lmbd)
            y_value += c*t - L
            if y_value < 0:
                loses_counter += 1
                break
    return loses_counter/attempts


probability(1, 1)
n = 10**6
times = [1, 3, 5]
for time in times:
    print(f"Prawdopodobieństwo niewypłacalności dla czasu {time}: ", probability(time, n))