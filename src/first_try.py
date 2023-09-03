import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable, List
import random as rand

"""
Proof-of-concept
"""

# Get the polynomial equation as a user input


# Define a symbolic variable

def simple_walk(data: List[np.float32]):
    res_i = []
    res_data = []
    MAX_I = len(data)
    i = rand.randint(0, MAX_I)

    while True:
        cur_v = data[i]

        tmp = data[i + 1] if i == 0 else (data[i - 1] if i == (MAX_I - 1) else min(data[i - 1], data[i + 1]))

        tmp = min(cur_v, tmp)

        if tmp == cur_v:
            break

        if tmp == data[i - 1]:
            i -= 1
        else:
            i += 1

        res_i.append(i)
        res_data.append(tmp)

    return res_i, res_data



def anim_func(f, data_i, data, line: plt.Line2D):
    line.set_data((data_i[:f],  data[:f]))

Writer = animation.writers['ffmpeg']
Writer = Writer(fps=10, metadata=dict(artist="Me"), bitrate=-1)

data = generate_datapoints(fun, 0.1)
res_i, res_data = simple_walk(data)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data)
ax2 = ax.twinx()
l, = ax.plot([], [], 'r-')
anim = animation.FuncAnimation(fig, anim_func, interval=50, fargs=(res_i, res_data, l), save_count=len(res_i))
anim.save("lines.mp4", writer=Writer)
