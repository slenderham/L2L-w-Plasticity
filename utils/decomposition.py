# -*- coding: utf-8 -*-

# import tensorly as tl
from tensorly.decomposition import parafac
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.style.use('seaborn-pastel');
colors1 = plt.cm.tab20b(np.linspace(0, 1, 128))
colors2 = plt.cm.tab20c(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

def forbenius_norm(mat, lag):

	# forbenius norm cross some lag

	return [np.trace(np.matmul(mat[i+lag].T, mat[i]))/\
			 np.trace(np.matmul(mat[i].T, mat[i]))/\
			 np.trace(np.matmul(mat[i+1].T, mat[i+1])) \
				 for i in range(len(mat)-lag-1)];

def decomp(vs, dUs, dims, chunks, turns):

	# dimension of the data should be num_samples X time_steps X channels
	# want it to be num_chunks X exposure X channels

	# for each sample, SORT it based on the order of dimension
	# average across samples, then we have time_steps X channels

	# figure out some ways to reshape such that all samples with the same first dimension are of the same task

	record = torch.load('WCST');
	vs = record['vs'];
	dUs = record['dUs'];
	dims = record['dims'];

	vs = np.array([[v.squeeze().detach().numpy() for v in v_sample] for v_sample in vs]);
	dUs = np.array([[dU.squeeze().detach().numpy() for dU in dU_sample] for dU_sample in dUs]);

	num_samples, time_steps, channel_num = vs.shape;

	dims = np.array(list(range(turns))*(time_steps//turns)).reshape(1, -1) \
					+ np.array(dims)*turns \
					+ np.concatenate([np.ones(turns*chunks)*chunks*turns*i for i in range(time_steps//chunks//turns)]).reshape(1, -1);

	for i in range(num_samples):
		vs[i, dims[i].astype(int)] = vs[i, :];
		dUs[i, dims[i].astype(int)] = dUs[i, :];

	vs = np.mean(vs, axis=0);
	dUs = np.mean(dUs, axis=0);

	vs_new = np.empty((chunks, time_steps//chunks, channel_num));
	dUs_new = np.empty((chunks, time_steps//chunks, channel_num, channel_num));

	for i in range(chunks):
		vs_new[i] = vs[np.concatenate([range(j*chunks*turns + i*turns, j*chunks*turns + (i+1)*turns) for j in range(time_steps//chunks//turns)])];
		dUs_new[i] = dUs[np.concatenate([range(j*chunks*turns + i*turns, j*chunks*turns + (i+1)*turns) for j in range(time_steps//chunks//turns)])];

	vFactors = parafac(vs_new, rank=1);
	plt_vs, axes = plt.subplots(1, 3);
	for i in range(3):
		axes[i].plot(vFactors[i]);

	dUsFactors = parafac(dUs_new.reshape(chunks, time_steps//chunks, -1), rank=1);
	plt_sUs, axes = plt.subplots(1, 3);
	for i in range(3):
		axes[i].plot(dUsFactors[i]);

	pca_v = PCA();
	low_v = pca_v.fit_transform(vs);
	axe =plt.figure().add_subplot(111, projection='3d')
	axe.scatter(low_v[:,0], low_v[:,1],low_v[:,2], cmap=mymap, c=np.arange(400))
	axe.plot(low_v[:,0], low_v[:,1], low_v[:,2],alpha=0.1, c='black')

	pca_dU = PCA();
	low_dU = pca_dU.fit_transform(dUs.reshape(400, -1));
	axe =plt.figure().add_subplot(111, projection='3d')
	axe.scatter(low_dU[:,0], low_dU[:,1],low_dU[:,2], cmap=mymap, c=np.arange(400))
	axe.plot(low_dU[:,0], low_dU[:,1], low_dU[:,2], alpha=0.1, c='black')

