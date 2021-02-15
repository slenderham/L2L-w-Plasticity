# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# fig, axes = plt.subplots(2, 5);
# ims = [];
# ax = plt.axes(xlim=(0, dUs.shape[-2]), ylim=(0, dUs.shape[-1]))
# for i in range(10):
#     ims.append(axes[i//5,i%5].imshow(dUs[i,500,:], cmap="twilight_r"));
fig, ax = plt.subplots(1)
im = plt.imshow(dUs[0,0].detach().squeeze(), cmap="seismic", vmin=-10, vmax=10);
plt.colorbar(im);

def init():
	# for i in range(10):
		# ims[i].set_data(dUs[i,500,:]);
	im.set_data(dUs[0,0,:].detach());
	return im;

def animate(i):
	# for j in range(10):
	# 	ims[j].set_data(dUs[j,i,:]);
	im.set_data(dUs[i,0,:].detach());
	return im;

anim = FuncAnimation(fig, animate, init_func=init, frames=300, interval=200, blit=False)
plt.show();
#anim.save('STDP.gif',writer='imagemagick')
#print("Done");
