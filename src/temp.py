import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable, List
import random as rand

"""
Proof-of-concept 
"""

A = 30
B = 20
C = 1

fun = lambda x: A*(x**2) + B * x + C

XBOUND_MIN = -30
XBOUND_MAX = 30

def simple_walk(data: List[np.float32]):
    res = []
    MAX_I = len(data)
    i = rand.randint(0, MAX_I)

    while True:
        cur_v = data[i]
        
        tmp = data[i+1] if i == 0 else (data[i-1] if i == (MAX_I-1) else min(data[i-1], data[i+1]))

        tmp = min(cur_v, tmp)
        
        if(tmp == cur_v):
            break
    
        res.append(tmp)
        if(tmp == data[i-1]): i -= 1
        else: i += 1

    return res

def generate_datapoints(function: Callable[[float], float], step: float = 1.0):
    temp = np.arange(XBOUND_MIN, XBOUND_MAX, step, dtype=np.float32)
    
    return list(map(function, temp))

def anim_func(f, datapts, line: plt.Line2D):
    line.set_data((range(f), datapts[:f]))


Writer = animation.writers['ffmpeg']
Writer = Writer(fps=10, metadata=dict(artist="Me"), bitrate=-1)

data = generate_datapoints(fun, 0.1)
res = simple_walk(data)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data)
ax2 = ax.twinx()
l, = ax.plot([], [], 'r-')
anim = animation.FuncAnimation(fig, anim_func, interval=50, fargs=(res, l))
anim.save("lines.mp4", writer=Writer)