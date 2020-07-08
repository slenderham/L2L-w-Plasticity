from numpy import linalg

import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure();
ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
sc = plt.scatter([],[]);

def init():
	sc.set_offsets([]);
	return sc,;

def animate(i):
	vecs = linalg.eigvals((model.rnns[0].h2h.weight.detach() \
						+ model.rnns[0].alpha.abs().detach()*dUs[0][i]).squeeze());
	sc.set_offsets(list(zip(vecs.real, vecs.imag)));
# 	if (i==0):
# 		print('cycle');
	return sc,;

anim = FuncAnimation(fig, animate, init_func=init, frames=400, interval=100, blit=True)
plt.show();
anim.save('../../figures/eigs.gif',writer='imagemagick')
#print("Done");