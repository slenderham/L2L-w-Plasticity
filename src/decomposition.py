# -*- coding: utf-8 -*-

# import tensorly as tl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import colorcet as cc

import numpy as np
import pickle
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tensortools as tt

plt.style.use('seaborn-pastel');
colors1 = plt.cm.tab20b(np.linspace(0, 1, 128))
colors2 = plt.cm.tab20c(np.linspace(0, 1, 128))

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

	vs = np.array([[v.squeeze().detach().numpy() for v in v_sample] for v_sample in vs]); # flatten 
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

def vis_parafac(x, rank):
    U = tt.cp_als(x, rank=rank, verbose=True)
    V = tt.cp_als(x, rank=rank, verbose=True)

    # Align the two fits and print a similarity score.
    sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
    print(sim)

    # make 
    if (x.shape==3):
        fig, axes = plt.subplots(rank, 3);
        for i in range(rank):
            axes[0, i].scatter(np.arange(len(U.factors[i][1])), U.factors[i][1])
            axes[0, i].set_xlabel("Time")
            axes[1, i].plot(U.factors[i][0])
            axes[1, i].set_xlabel("Trial")
            axes[2, i].bar(U.factors[i][2]);
            axes[2, i].set_xlabel("Neuron")
    elif (x.shape==4):
        fig, axes = plt.subplots(rank, 4);
        for i in range(rank):
            axes[0, i].scatter(np.arange(len(U.factors[i][1])), U.factors[i][1])
            axes[0, i].set_xlabel("Time")
            axes[1, i].plot(U.factors[i][0])
            axes[1, i].set_xlabel("Trial")
            mat_factor = np.outer(U.factors[i][2], U.factors[i][3])
            lim = max(max(mat_factor), -min(mat_factor));
            axes[2, i].imshow(mat_factor, cmap='coolwarm', vmin=-lim, vmax=lim);
            axes[2, i].set_xlabel("Presynaptic Neuron")
            axes[2, i].set_ylabel("Postsynaptic Neuron")
    else:
        raise ValueError('shape of tensor x needs to be 3 (batch time series of vectors) or 4 (batch time series of matrices');

    fig.suptitle("")
    fig.tight_layout()
    plt.show();

def vis_pca(x, tags):
    assert(len(x.shape)==2);
    assert(len(tags.shape)==1);
    assert(x.shape[0]==tags.shape[0]);
    cmap = mcolors.LinearSegmentedColormap.from_list('mymap', cc.glasbey_light[:22]);
    pca = PCA();
	low_x = pca.fit_transform(x);
	axe = plt.figure().add_subplot(111, projection='3d')
	axe.scatter(low_x[:,0], low_x[:,1],low_x[:,2], cmap=cmap, c=tags);
	axe.plot(low_x[:,0], low_x[:,1], low_x[:,2],alpha=0.1, c='black');
    plt.figure().tight_layout();
    plt.show();