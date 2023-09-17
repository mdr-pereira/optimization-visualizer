import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import constants as _c

"""
We want to write all of the visualization functions in here, receiving the data from the main loop over at the "main.py" file.
"""

def buildCurrentText(f, function, optimal_y, agg_x):
    return f"""Current Values\n
                \n\nIteration:\n {f}
                \n\nPosition:\n ({round(agg_x[f], 3)}, {round(function(agg_x[f]), 2)})
                \n\nLoss:\n {round(1/2*(optimal_y - function(agg_x[f]))**2, 2)}"""

def anim_func(f, function, derivative, optimal_y, agg_x, points, ax1, ax2, ax_n):
    #Update the line with coordinates up to the current point (frame)
    ax_n.set_title(f"t = [{f}]", loc="center", fontweight = "bold", fontsize=12)
    ax_n.texts[0].set_text(buildCurrentText(f, function, optimal_y, agg_x))

    #Add a new point to the line
    points.append(ax1.plot(agg_x[f], function(agg_x[f]), 'ro', markersize=6))
    points.append(ax2.plot(agg_x[f], derivative(agg_x[f]), 'ro', markersize=6))
   
    #Paint the points in the line in a gradient of colors
    for i in range(len(points)):
        points[i][0].set_color(plt.cm.viridis(i/len(points)))

    return points


def build_writer():
    writer = animation.writers['ffmpeg']
    writer = writer(fps=10, metadata=dict(artist="Visualization Group: Manuel Pereira, Surendhar, Mawuko Tettey"), bitrate=-1)

    return writer

def plot(optimizer_name, equation, function, derivative, res_x):
    print("Plotting...")

    writer = build_writer()

    x = np.arange(_c.XBOUND_MIN, _c.XBOUND_MAX, 0.1, dtype=np.float32)

    #Create a figure with two plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
   
    ax1.plot(x, function(x), 'b-')
    ax_n = ax1.twinx()

    ax_n.set_yticks([])
    ax_n.set_yticklabels([])
    ax_n.set_xticks([])
    ax_n.set_xticklabels([])

    #Add a new plot at the bottom of the figure
    ax2.plot(x, derivative(x), 'b-')

    #Add a horizontal line at y=0
    ax2.axhline(y=0, color='b', linestyle='--')

    #Separate both plots completely
    plt.subplots_adjust(hspace=0.5)

    fig.suptitle(f"{optimizer_name}\n\n{str(equation)}", fontweight="bold", fontsize=14)    

    #Add title to figure 
    ax1.set_title("Solution Space", loc="left")
    ax2.set_title("Solution Space Derivative", loc="left")

    data = [function(_x) for _x in res_x]

    optimal_x = round(res_x[np.argmin(data)], 3)
    optimal_y = round(np.min(data), 3)

    optimal_text = f"""Optimal Values\n
                    \nPosition:\n ({optimal_x},{optimal_y})"""
    
    final_x = round(res_x[-1], 3)
    final_y = round(function(res_x[-1]), 3)
    
    solution_text = f"""Solution Values\n
                \n\nTotal Iterations:\n {round(len(res_x), 3)}
                \n\nInitial Position:\n ({round(res_x[0],3)}, {round(function(res_x[0]), 3)})
                \n\nFinal Position:\n ({final_x}, {final_y})
                \n\nFinal Loss:\n {round(1/2*(optimal_y - final_y)**2, 3)}"""

    props = dict(boxstyle='round', facecolor='bisque', alpha=0.15)  # bbox features

    #Add multiple textboxes outside of the figure, each with their own properties. Make sure they are fully separated from the figure and eachother
    ax1.text(1.02, 0.7, optimal_text, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
    ax2.text(1.02, -0.15, solution_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    ax_n.text(1.02, 0.6, "", transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
   
    #Make both textboxes the same size
    ax1.texts[0].set_bbox(dict(facecolor='bisque', alpha=0.15))
    ax2.texts[0].set_bbox(dict(facecolor='bisque', alpha=0.15))
    ax_n.texts[0].set_bbox(dict(facecolor='bisque', alpha=0.15))

    points = []

    fig.tight_layout()

    #Give a little bit of space to the top of the figure
    fig.subplots_adjust(top=0.85)


    #Add a red vertical line at the optimal value
    ax1.axvline(x=x[np.argmin([function(_x) for _x in x])], color='r', linestyle='--')    

    anim = animation.FuncAnimation(fig, anim_func, interval=50, fargs=(function, derivative, optimal_y, res_x, points, ax1, ax2, ax_n), save_count=len(res_x))
    anim.save("lines.mp4", writer=writer)
