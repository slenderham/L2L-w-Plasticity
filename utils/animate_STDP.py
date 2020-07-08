# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


fig, axes = plt.subplots(2, 5);
ims = [];
#ax = plt.axes(xlim=(0, dUs.shape[0]+1), ylim=(0, dUs.shape[1]+1))
# for i in range(10):
#     ims.append(axes[i//5,i%5].imshow(dUs[i,500,:], cmap="twilight_r"));
# im = plt.imshow(dUs[1].detach().squeeze(), cmap="seismic");
# plt.colorbar(im);

def init():
	# for i in range(10):
		# ims[i].set_data(dUs[i,500,:]);
	imsset_data(dUs[0,0,:]);
	return ims;

def animate(i):
	# for j in range(10):
	# 	ims[j].set_data(dUs[j,i,:]);
	ims.set_data(dUs[0,i,:]);
	return ims;

anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=1, blit=False)
plt.show();
#anim.save('STDP.gif',writer='imagemagick')
#print("Done");
