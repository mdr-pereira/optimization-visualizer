import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
We want to write all of the visualization functions in here, receiving the data from the main loop over at the "main.py" file.
"""


def anim_func(f, data_i, data, line: plt.Line2D):
    line.set_data((data_i[:f],  data[:f]))


def plot(res_i, res_data, data):
    Writer = animation.writers['ffmpeg']
    Writer = Writer(fps=10, metadata=dict(artist="Me"), bitrate=-1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data)
    ax2 = ax.twinx()
    l, = ax.plot([], [], 'r-')
    anim = animation.FuncAnimation(fig, anim_func, interval=50, fargs=(res_i, res_data, l), save_count=len(res_i))
    anim.save("lines.mp4", writer=Writer)
