#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:07:28 2019

@author: wangchong
"""

import torch

def sigma_inv(x):
    return torch.log(x)-torch.log(1.0-x);

class NormalizeWeight(torch.nn.Module):
	def __init__(self):
		super(NormalizeWeight, self).__init__();

	def forward(self, x):
		x = torch.relu(x);
		return x/(torch.sum(x, dim=1, keepdim=True)+1e-8);

class kWTA(torch.nn.Module):
	def __init__(self, sparsity=0.2):
		super(kWTA, self).__init__();
		assert (sparsity<=1.0 and sparsity>0.0), "please enter sparsity 0.0<s<=1.0"
		self.sparsity = sparsity;

	def forward(self, x):
		k = round(x.shape[1]*self.sparsity);
		assert (round(1/self.sparsity)>1), "no units are active after rounding! please increase sparsity"
		thresh = torch.topk(x, k, dim=1)[0][:,-2:-1]+1e-8;
		x = torch.relu(x-thresh);
		return x/(torch.sum(x, dim=1, keepdim=True)+1e-8);

class BipolarWrapper(torch.nn.Module):
	def __init__(self, module, positive_proportion=0.8):
		super(BReLU, self).__init__();
		self.module = module;
		assert positive_proportion>0 and positive_proportion<=1, "please enter a valid proportion"
		self.positive_proportion = positive_proportion;

	def forward(self, x):
		cutoff = round(x.shape[1]*self.positive_proportion);
		result = torch.empty_like(x);
		result[:,:cutoff] = self.module.forward(x[:cutoff])
		result[:,cutoff:] = -self.module.forward(-x[cutoff:]);
		return result;

class SaturatingPoisson(torch.nn.Module):
	def __init__(self):
		super(SaturatingPoisson, self).__init__();

	def forward(self, x):
# 		return 1-torch.exp(-torch.relu(x));
		return (x>0)*(1-torch.exp(-torch.relu(x/2))) + (x<=0)*x*torch.sigmoid(x);

class Swish(torch.nn.Module):
	def __init__(self):
		super(Swish, self).__init__();

	def forward(self, x):
		return x*torch.sigmoid(x);


class Spike(torch.nn.Module):
	def __init__(self, gamma=0.3):

		# the spike activation function that uses the surrogate gradient from

		super(Spike, self).__init__();
		self.gamma = gamma;

	def forward(self, x, thresh):
		surr = torch.nn.functional.hardtanh(x) * (1 + torch.abs(x)/2);
		spikes = x>0;

		# use gamma to scale the gradient, but doesn't affect the forward output

		return spikes + self.gamma*(surr.detach() - surr);

class LeakyIntegrate(torch.nn.Module):
	def __init__(self, dt, input_dim, out_dim):
		super(LeakyIntegrate, self).__init__();
		self.dt = dt;
		self.decoder = torch.nn.Linear(input_dim, out_dim);

	def forward(self, prev, x):
		return prev + self.dt*self.decoder(x);

class TernaryTanh(torch.nn.Module):
	def __init__(self):
		super(TernaryTanh, self).__init__();

	def forward(self, x):
		return torch.tanh(x**3/3.0);

# class LogisticSpike(torch.nn.Module):
# 	def __init__(self):
# 		base_distribution = Uniform(0, 1)
# 		transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
# 		self.logistic = TransformedDistribution(base_distribution, transforms);

# 	def forward(self, x):
# 		noise = self.logistic.sample(x.shape);

