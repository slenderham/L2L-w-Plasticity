# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# fig, axes = plt.subplots(2, 5);
# ims = [];
# ax = plt.axes(xlim=(0, dUs.shape[-2]), ylim=(0, dUs.shape[-1]))
# for i in range(10):
#     ims.append(axes[i//5,i%5].imshow(dUs[i,500,:], cmap="twilight_r"));

def animate_traj(x, labels):

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    sct = ax.scatter([0], [0], [0])
    ax.set_xlim3d([-30, 30])
    ax.set_xlabel('X')

    ax.set_ylim3d([-30, 30])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-30, 30])
    ax.set_zlabel('Z')

    def init():
        sct._offsets3d = ([0], [0], [0])
        return sct;

    def animate(i):
        print(i)
        # for j in range(10):
        # 	ims[j].set_data(dUs[j,i,:]);
        sct._offsets3d = (x[i,0:1], x[i,1:2], x[i,2:3])
        # sct.set_color(colors[labels[i]])
        return sct;
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=300, interval=200, blit=True)
    plt.show();