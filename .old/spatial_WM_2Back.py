# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from modulated_full import SGRU
import torch.optim as optim
from scipy.stats import ortho_group
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from utils.lamb import Lamb
from tqdm import tqdm

'''
	spatial WM task
	sequential MNIST addition task
'''

torch.manual_seed(0);

def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if ("ln" in name or "encoder" in name or "weight" not in name):
			no_decay.append(param);
		else:
			decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-3}];

def sequentialize(batch, noise_dur, noise_scale, grace_period):
	'''
		given a batch of 2N * C * H * W,
		transform into a sequence of size 2*(H+(random noise)) * N * W

	'''
	img, label = batch;


	img = img.squeeze().transpose(0, 1); # H * N * W
	img = torch.cat((\
						img[:,:label.shape[0]//2,:], \
						torch.randn((noise_dur, img.shape[1]//2, img.shape[2]))*noise_scale, \
						img[:,label.shape[0]//2:,:], \
						torch.randn((noise_dur, img.shape[1]//2, img.shape[2]))*noise_scale, \
						torch.zeros((grace_period, img.shape[1]//2, img.shape[2]))),\
					dim=0);

	label = label[:label.shape[0]//2] + label[label.shape[0]//2:];

	return img.to(device), label.to(device);

n_epochs = 13;
batch_size = 50;
drawInt = 100;
noise_dur = 10;
grace_period = 10;
noise_scale = 1;

AR = 0.02;
TAR = 0.02;

mnist_data_train = datasets.MNIST(root="./data", train=True, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)),
				   ]), download = True);

train_iter = torch.utils.data.DataLoader(mnist_data_train, batch_size=batch_size*2, shuffle=True);


mnist_data_test = datasets.MNIST(root="./data", train=False, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)),
				   ]), download = True);

test_iter = torch.utils.data.DataLoader(mnist_data_test, batch_size=batch_size*2, shuffle=True);



model = SGRU(in_type = "continuous",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = 28,\
			 hidden_dim = 256,\
			 out_dim = 20,\
			 num_layers = 2,\
			 activation="swish",\
			 mod_rank = 16,\
			 );

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device);
else:
	device = torch.device("cpu");

param_groups = add_weight_decay(model);

optimizer = Lamb(param_groups, lr=1e-3);
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5);

criterion = torch.nn.NLLLoss();

loss = 0;


for i in range(n_epochs):
	dUs = [];
	for idx, batch in enumerate(tqdm(train_iter)):
		new_h, new_v, new_dU, new_trace = model.train().get_init_states(batch_size=batch_size, device=device);

		img, label = sequentialize(batch, noise_dur, noise_scale, grace_period);

		new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.forward(\
													  x = img,\
													  h = new_h, \
													  v = new_v, \
													  dU = new_dU, \
													  trace = new_trace);

		loss = 0.5*criterion(output[:output.shape[0]-noise_dur, :], 20*torch.oness_like(output[:output.shape[0]-noise_dur, :], dtype=int));
		loss += criterion(output[-1, :], label);
		loss += AR*(last_layer_out.pow(2).mean());
		loss += TAR*(last_layer_out[1:]-last_layer_out[:-1]).pow(2).mean();

		loss.backward();

		if idx%100==0: print(loss);
		torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1);
		optimizer.step();
		optimizer.zero_grad();loss=0;

	error = 0;

	with torch.no_grad():
		for idx, batch in enumerate(tqdm(test_iter)):
			new_h, new_v, new_dU, new_trace = model.eval().get_init_states(batch_size=batch_size, device=device);

			img, label = sequentialize(batch, noise_dur, noise_scale, grace_period);

			new_v, new_h, new_dU, new_trace, output = model.forward(\
								  x = img,\
								  h = new_h, \
								  v = new_v, \
								  dU = new_dU, \
								  trace = new_trace);

			error += 1.0*torch.sum(torch.argmax(output[-1,:,:], dim=1)!=label)/5000.0;

	print(error);
	scheduler1.step();

	torch.save({'model_state_dict': model.state_dict()}, 'model_PTB');