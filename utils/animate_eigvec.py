#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:17:03 2019

@author: wangchong
"""
'''
can't visualize high D eigenvectors, I was delusional

'''
# -*- coding: utf-8 -*-
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.gca(projection='3d')
quiv = ax.quiver([],[],[],[],[],[])

def init():
	quiv.set_data([],[],[],[],[],[]);
	return quiv,;

def animate(i):
	eigenVal, eigenVec = linalg.eig(dUs[:,:,i].detach());
	idx = eigenVal.argsort()[::-1]
	eigenVal = eigenVal[idx]
	eigenVec = eigenVec[:,idx[:3]]
	quiv.set_data([0,0],[0],[0],)

	return sc,;


anim = FuncAnimation(fig, animate, init_func=init, frames=dUs.shape[2], interval=20, blit=True)
plt.show();
#anim.save('eigens.gif',writer='imagemagick')
#print("Done");
