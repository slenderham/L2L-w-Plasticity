# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from modulated_ctRNN import SGRU
import torch.optim as optim
from scipy.stats import ortho_group
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb
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

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-5}];

def sequentialize(batch, noise_dur, noise_scale, grace_period):
	'''
		given a batch of N * C * H * W,
		transform into a sequence of size H+(random noise) * N * W

	'''
	img, label = batch;


	img = img.squeeze().flatten(start_dim=1).t().unsqueeze(2); # N * (HXW) * 1
# 	img = torch.cat((\
# 						img, \
# 						torch.randn((noise_dur, img.shape[1], img.shape[2]))*noise_scale, \
# 						torch.zeros((grace_period, img.shape[1], img.shape[2]))),\
# 					dim=0);

	return img.to(device), label.to(device);

n_epochs = 12;
batch_size = 100;
drawInt = 100;
noise_dur = 0;
grace_period = 0;
noise_scale = 1;

AR = 1e-3;
TAR = 1e-3;

mnist_data_train = datasets.MNIST(root="./data", train=True, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)),
				   ]), download = True);

train_iter = torch.utils.data.DataLoader(mnist_data_train, batch_size=batch_size, shuffle=True);


mnist_data_test = datasets.MNIST(root="./data", train=False, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)),
				   ]), download = True);

test_iter = torch.utils.data.DataLoader(mnist_data_test, batch_size=batch_size, shuffle=True);


model = SGRU(in_type = "continuous",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = 1,\
			 hidden_dim = 80,\
			 out_dim = 11,\
			 num_layers = 1,\
			 activation="relu",\
			 mod_rank = 20,\
			 );print(model);

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device);
else:
	device = torch.device("cpu");

param_groups = add_weight_decay(model);

optimizer = Lamb(param_groups, lr=4e-3, min_trust=1e-3);
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8);

criterion = torch.nn.NLLLoss();

loss = 0;

# try:
# 	state_dict = torch.load("model_SMNIST");
# 	model.load_state_dict(state_dict["model_state_dict"]);
# 	optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
# except:
# 	None;

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

		loss = criterion(output[:output.shape[0]-grace_period, :].view(-1, 11), 10*torch.ones((output.shape[0]-grace_period)*batch_size, dtype=int, device=device));
		loss += criterion(output[-1, :], label);
		loss += AR*(last_layer_out.pow(2).mean());
		loss += TAR*(last_layer_out[1:]-last_layer_out[:-1]).pow(2).mean();

		loss.backward();model.scale_grad();

		if idx%10==0:
			print(loss);
		torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5);
		optimizer.step();
		optimizer.zero_grad();loss=0;

	error = 0;

	with torch.no_grad():
		for idx, batch in enumerate(tqdm(test_iter)):
			new_h, new_v, new_dU, new_trace = model.eval().get_init_states(batch_size=batch_size, device=device);

			img, label = sequentialize(batch, noise_dur, noise_scale, grace_period);

			new_v, new_h, new_dU, new_trace, (_, output) = model.forward(\
								  x = img,\
								  h = new_h, \
								  v = new_v, \
								  dU = new_dU, \
								  trace = new_trace);

			error += 1.0*torch.sum(torch.argmax(output[-1,:,:], dim=1)!=label)/10000.0;

	print(error);
	scheduler1.step();

	torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model_SMNIST');