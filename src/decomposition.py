# -*- coding: utf-8 -*-

# import tensorly as tl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import colorcet as cc

import torch
import numpy as np
import pickle
import os
# import tensortools as tt
from tensorly.decomposition import parafac, tucker
from tensorly.regression import tucker_regression

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.cross_decomposition import CCA
from sklearn import metrics
from sklearn.model_selection import cross_val_score

plt.style.use('seaborn-pastel');
colors1 = plt.cm.tab20b(np.linspace(0, 1, 128))
colors2 = plt.cm.tab20c(np.linspace(0, 1, 128))

def forbenius_norm(mat, lag):
    # forbenius norm cross some lag
    # return [np.trace(np.matmul(mat[i+lag].T, mat[i]))/\
    #                     np.trace(np.matmul(mat[i].T, mat[i]))/\
    #                     np.trace(np.matmul(mat[i+1].T, mat[i+1])) \
    #                             for i in range(len(mat)-lag-1)];
    return [np.mean((mat[i+lag]-mat[i])**2) for i in range(len(mat)-lag-1)];

def decomp(vs, dUs, dims, chunks, turns):

        # dimension of the data should be num_samples X time_steps X channels
        # want it to be num_chunks X exposure X channels

        # for each sample, SORT it based on the order of dimension
        # average across samples, then we have time_steps X channels

        # figure out some ways to reshape such that all samples with the same first dimension are of the same task

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

def calculateAttention(mem, query, history):
    value = np.matmul(mem, np.expand_dims(query, -1)).squeeze();
    # value should be a linear combination of previous states -  Hw = Mq, try to find w
    # Do linear regression! w = (H'H)^-1H'Mq
    w = [];
    for i in range(mem.shape[0]):
        lin_reg = LinearRegression(fit_intercept=False).fit(history[:,i].T, np.expand_dims(value[i],-1))
        w.append(lin_reg.coef_);

    return w;

def vis_parafac(x, rank, plot_type):
    # ens = tt.Ensemble()
    # Align the two fits and print a similarity score.
    # ens.fit(x, ranks=list(range(1, rank)), replicates=2, verbose=True)

    # fig, axes = plt.subplots(1, 2);

    # tt.visualization.plot_objective(ens, ax=axes[0])
    # tt.visualization.plot_similarity(ens, ax=axes[1]);
    assert(plot_type in ['omni_vec', 'omni_mat', 'wcst_vec', 'wcst_mat'])
    # U = tt.cp_als(x, rank=rank)
    U = tucker(x, rank=rank, verbose=True)

    if (plot_type=='omni_vec'):
        fig, axes = plt.subplots(3, rank);
        for i in range(rank):
            axes[0, i].scatter(np.arange(len(U.factors[1][:,i])), U.factors[1][:,i])
            axes[0, i].set_xlabel("Trial", fontsize=5)
            axes[1, i].plot(U.factors[0][:,i])
            axes[1, i].set_xlabel("Time", fontsize=5)
            axes[2, i].bar(np.arange(len(U.factors[2][:,i])), U.factors[2][:,i]);
            axes[2, i].set_xlabel("Neuron", fontsize=5)
    elif (plot_type=='omni_mat'):
        fig, axes = plt.subplots(3, rank);
        for i in range(rank):
            axes[0, i].scatter(np.arange(len(U.factors[1][:,i])), U.factors[1][:,i])
            axes[0, i].set_xlabel("Trial", fontsize=5)
            axes[1, i].plot(U.factors[0][:,i])
            axes[1, i].set_xlabel("Time", fontsize=5)
            mat_factor = np.outer(U.factors[2][:,i], U.factors[3][:,i])
            # lim = max(np.max(mat_factor), -np.min(mat_factor));
            im = axes[2, i].imshow(mat_factor, cmap='seismic');
            axes[2, i].set_xlabel("Presynaptic Neuron", fontsize=5)
            axes[2, i].set_ylabel("Postsynaptic Neuron", fontsize=5)
            fig.colorbar(im, ax=axes[2, i])
    elif (plot_type=='wcst_vec'):
        fig, axes = plt.subplots(4, rank);
        for i in range(rank):
            axes[0, i].plot(U.factors[1][:,i])
            axes[0, i].set_xlabel("Inter-episode Trial", fontsize=5)
            axes[1, i].scatter(np.arange(len(U.factors[0][:,i])), U.factors[0][:,i])
            axes[1, i].set_xlabel("Time", fontsize=5)
            axes[2, i].scatter(np.arange(len(U.factors[2][:,i])), U.factors[2][:,i])
            axes[2, i].set_xlabel("Episodes", fontsize=5)
            axes[3, i].bar(np.arange(len(U.factors[3][:,i])), U.factors[3][:,i]);
            axes[3, i].set_xlabel("Neuron", fontsize=5)
    elif (plot_type=='wcst_mat'):
        fig, axes = plt.subplots(4, rank);
        for i in range(rank):
            axes[0, i].plot(U.factors[1][:,i])
            axes[0, i].set_xlabel("Inter-episode Trial", fontsize=5)
            axes[1, i].scatter(np.arange(len(U.factors[0][:,i])), U.factors[0][:,i])
            axes[1, i].set_xlabel("Time", fontsize=5)
            axes[2, i].scatter(np.arange(len(U.factors[2][:,i])), U.factors[2][:,i])
            axes[2, i].set_xlabel("Episodes", fontsize=5)
            mat_factor = np.outer(U.factors[3][:,i], U.factors[4][:,i])
            # lim = max(np.max(mat_factor), -np.min(mat_factor));
            im = axes[3, i].imshow(mat_factor, cmap='seismic');
            axes[3, i].set_xlabel("Presynaptic Neuron", fontsize=5)
            axes[3, i].set_ylabel("Postsynaptic Neuron", fontsize=5)
            fig.colorbar(im, ax=axes[3, i])
    else:
        raise ValueError('shape of tensor x needs to be 3 (batch time series of vectors) or 4 (batch time series of matrices');

    fig.suptitle("")
    fig.tight_layout()
    plt.show()
    return U

def vis_pca(x, labels=None, tags=None, threeD=False, data_type='vec'):
    # assert(len(x.shape)==3);
    assert(len(tags.shape)==2);
    x_unflat_shape = x.shape[:2]
    tags_unflat = tags.copy()
    x = x.flatten(0, 1).numpy()
    tags = tags.flatten()
    assert(x.shape[0]==tags.shape[0]);
    if data_type=='mat':
        U = tucker(x, rank=3, verbose=True)
        low_x = U.factors[0]
    elif data_type=='vec':
        pca = PCA()
        low_x = pca.fit_transform(x);
    else:
        raise ValueError
    
    bounds = np.linspace(0, plt.get_cmap('tab10').N, plt.get_cmap('tab10').N+1)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=plt.get_cmap('tab10').N)
    
    if threeD:
        axe = plt.figure().add_subplot(111, projection='3d')
        scatters = []
        for i in range(tags.max()+1):
            scatters.append(axe.scatter(low_x[tags==i][:,0], low_x[tags==i][:,1], low_x[tags==i][:,2], c=tags[tags==i], label=labels[i], norm=norm, cmap='tab10'));
        # legend = axe.legend(scatters, labels)
    else:
        axe = plt.figure().add_subplot(111)
        scatters = []
        for i in range(tags.max()+1):
            scatters.append(axe.scatter(low_x[tags==i][:,0], low_x[tags==i][:,1], c=tags[tags==i], label=labels[i], norm=norm, cmap='tab10'));
        # legend = axe.legend(scatters, labels)
    
    low_x = low_x.reshape(*x_unflat_shape, -1)
    
    if threeD:
        segments = np.concatenate([np.expand_dims(low_x[:-1,:5,:3], 2), np.expand_dims(low_x[1:,:5,:3], 2)], axis=2)
        segments = segments.reshape((segments.shape[0]*segments.shape[1], 2, 3))
        lc = Line3DCollection(segments, cmap='tab10', norm=norm, alpha=1)
        lc.set_array(tags)
        axe.add_collection(lc)
    else:
        segments = np.concatenate([np.expand_dims(low_x[:-1,:5,:2], 2), np.expand_dims(low_x[1:,:5,:2], 2)], axis=2)
        segments = segments.reshape((segments.shape[0]*segments.shape[1], 2, 2))
        print(segments.shape)
        lc = LineCollection(segments, cmap='tab10', norm=norm, alpha=0.2)
        lc.set_array(tags)
        axe.add_collection(lc)
    
    axe.set_xlabel('PC1')
    axe.set_ylabel('PC2')
    if threeD:
        axe.set_zlabel('PC3')
    # axe.add_artist(legend)
    plt.figure().tight_layout();
    plt.show();
    return axe;

def vis_lda(x, tags):
    lda = LinearDiscriminantAnalysis(n_components=min(tags.max(), 2));
    low_x = lda.fit_transform(x, tags);
    axe = plt.figure().add_subplot(111)
    bounds = np.linspace(0, plt.get_cmap('tab10').N, plt.get_cmap('tab10').N+1)
    if tags.max()>=2:
        scatters = []
        for i in range(tags.max()+1):
            scatters.append(axe.scatter(low_x[tags==i][:,0], low_x[tags==i][:,1], c=plt.get_cmap('tab10').colors[i], alpha=0.5));
        legend = axe.legend(scatters, tags, title="Task Type");
        axe.add_artist(legend)
        plt.figure().tight_layout();
        axe.set_xlabel('LD1')
        axe.set_ylabel('LD2')
    elif tags.max()==1:
        axe.hist(low_x[tags==0].flatten(), density=True, alpha=0.5, label='0')
        axe.hist(low_x[tags==1].flatten(), density=True, alpha=0.5, label='1')
        plt.legend()
        plt.figure().tight_layout();
        axe.set_ylim(0, 1)
        axe.set_xlabel('LD')
        axe.set_ylabel('Density')
    return axe;

def svc_cv(x, y, type='vec'):
    if type=='vec':
        clf = LinearSVC();
        scores = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
    else:
        idx = np.random.permutation(np.arange(x.shape[0]))
        num_samples_per_fold = x.shape[0]//5
        scores = np.zeros(x.shape[0])
        for i in range(5):
            clf = tucker_regression.TuckerRegressor(1)
            fold_idx = idx[num_samples_per_fold*i:num_samples_per_fold*(i+1)]
            clf.fit(x[fold_idx], y[fold_idx])
            # new_pred = 
        
    return scores;

def sig2asterisk(p):
    if p>0.05:
        return '    '
    elif p>0.01:
        return '  * '
    elif p>0.001:
        return ' ** '
    elif p>0.0001:
        return ' ***'
    else:
        return '****'