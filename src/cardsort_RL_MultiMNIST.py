import torch
import os
from modulated_AC import SGRU
import torch.optim as optim
from scipy.stats import ortho_group
from torchvision import datasets, transforms
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb
from a2c import compute_loss
import tqdm

torch.manual_seed(0);


def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if ("ln" in name or "encoder" in name or "weight" not in name):
			no_decay.append(param);
		else:
			decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-5}];

'''
make dataset
'''

root_dir='./dataset/triple_mnist';
num_image_per_class=1000;
num_digit=2;
image_size=32;

n_epochs = 30000;
seq_len = 8;
change_rule_period = 10;
val_len = 8;
batch_size = 40;
num_classes = 10;

# dataset
train_set = datasets.ImageFolder(root=os.path.join(root_dir, "train"), transform=transforms.Compose([
						transforms.Grayscale(),
						transforms.ToTensor(),
						transforms.Normalize((0.5,), (0.5,)),
				   ]));

val_set = datasets.ImageFolder(root=os.path.join(root_dir, "val"), transform=transforms.Compose([
						transforms.Grayscale(),
						transforms.ToTensor(),
						transforms.Normalize((0.5,), (0.5,)),
				   ]));
test_set = datasets.ImageFolder(root=os.path.join(root_dir, "test"), transform=transforms.Compose([
						transforms.Grayscale(),
						transforms.ToTensor(),
						transforms.Normalize((0.5,), (0.5,)),
				   ]));

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size*seq_len, shuffle=True);
val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size*seq_len, shuffle=True);
test_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size*seq_len, shuffle=True);

'''
make model
'''

beta_v = 0.5;
beta_a = 5e-3;

AR = 1e-2;
TAR = 1e-2;

drawInt = 1;

model = SGRU(in_type = "image",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = image_size+num_classes+1,\
			 hidden_dim = 256,\
			 out_dim = num_classes,\
			 num_layers = 1,\
			 activation="relu",\
			 mod_rank = 128\
			 );

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device);
else:
	device = torch.device("cpu");

param_groups = add_weight_decay(model);

# optimizer = Lamb(param_groups, lr=4e-3, min_trust=0.25);
optimizer = optim.AdamW(param_groups, lr=1e-3);
# optimizer = optim.SGD(param_groups, lr=1e-2, momentum=0.9);
# optimizer = optim.RMSprop(param_groups, lr=5e-4);
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9);

loss = 0;

for i in range(n_epochs):
	for idx, batch in enumerate(tqdm.tqdm(train_iter)):
		imgs, labels = batch;

		# img shape should be (batch_size*seq_len) X channel X row X column
		# squeeze and permute into row X (batch_size*seq_len) X column
		# chunked into [seq_len] lists of row X batch_size X column

		imgs = torch.chunk(imgs.to(device).squeeze().permute(2,0,1), chunks=seq_len, dim=1);
		labels = torch.chunk(labels.to(device), chunks=seq_len);

		# initialize loss and reward
		loss_reg = 0;
		reward = torch.zeros(batch_size, 1, device=device);
		new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
		action = torch.zeros(batch_size, num_classes, device=device);

		log_prob_of_act = [];
		values = [];
		rewards = [];

		newDim = torch.randint(low=0, high=num_digit, size=(batch_size,)).to(device);

		for jdx, (img, label) in enumerate(zip(imgs, labels)):

			if ((jdx)%change_rule_period==0):
				newDim = torch.randint(low=0, high=num_digit, size=(batch_size,)).to(device);



			# RNN works with size seq len X batch size X input size, in this case # chunks X 1 X pattern size + |A| + 1
			patterns = torch.cat(
									(img, torch.zeros((image_size, batch_size, num_classes+1), device=device)), dim=2
								);
			# feedback from previous trial, 1 X batch size X [0]*pattern_size + previous action + previous reward
			feedback = torch.cat(
									(torch.zeros((batch_size, image_size), device=device), 5*action.detach(), 5*reward.detach()), dim=1
								).reshape(1, batch_size, image_size+num_classes+1);

			total_input = torch.cat(
									(feedback, patterns), dim=0
								);

			# one iter of network, notice that the reward is from the previous time step
			new_v, new_h, new_dU, new_trace, (last_layer_out, log_probs, value) = model.train().forward(\
														  x = total_input.to(device),\
														  h = new_h, \
														  v = new_v, \
														  dU = new_dU, \
														  trace = new_trace);

			# sample an action
			m = torch.distributions.Categorical(logits = log_probs[-1]);
			action_idx = m.sample();
			action = torch.zeros(batch_size, num_classes, dtype=torch.float, device=device);
			action[torch.arange(batch_size), action_idx] = 1.0;

			# calculate entropy loss
			loss_reg += beta_a*torch.sum(log_probs[-1]*torch.exp(log_probs[-1]))+ AR*(last_layer_out.pow(2).mean()) + TAR*(last_layer_out[1:]-last_layer_out[:-1]).pow(2).mean();

			# get reward
			reward = 2*torch.as_tensor((((label//torch.pow(10, newDim)%10)%num_classes).flatten()==action_idx), dtype=torch.float, device=device).reshape(batch_size,-1).detach()-1;

			# save reward, log_prob, value;
			rewards.append(reward);
			log_prob_of_act.append(m.log_prob(action_idx));
			values.append(value[-1]);

		loss = compute_loss(rewards=torch.stack(rewards), \
							 values=torch.stack(values), \
							 log_probs=torch.stack(log_prob_of_act), \
							 gamma=0.9, \
							 N=20, \
							 beta_v=beta_v, \
						 )\
				+ loss_reg/seq_len/batch_size;

		loss.backward();
		# model.scale_grad();
		if (idx%10==0):
			print(loss, (seq_len*batch_size+torch.sum(torch.stack(rewards)))/2/seq_len/batch_size);
		torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0);
		optimizer.step();
		optimizer.zero_grad();
		if (idx!=0 and idx%drawInt==0):
			# beta_a = max(0.001, beta_a*0.99);
			scheduler1.step();

	dUs = [];
	vs = [];
	dims = [];
	cumReward = [];

	with torch.no_grad():
		for idx, batch in enumerate(tqdm.tqdm(val_iter)):
			imgs, labels = batch;

			# img shape should be (batch_size*seq_len) X channel X row X column
			# squeeze and permute into row X (batch_size*seq_len) X column
			# chunked into [seq_len] lists of row X batch_size X column

			imgs = torch.chunk(imgs.to(device).squeeze().permute(2,0,1), chunks=seq_len, dim=1);
			labels = torch.chunk(labels.to(device), chunks=seq_len);

			reward = torch.zeros(batch_size, 1, device=device);
			new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
			action = torch.zeros(batch_size, num_classes, device=device);

			newDim = torch.randint(low=0, high=num_digit, size=(batch_size,)).to(device);

			for jdx, (img, label) in enumerate(zip(imgs, labels)):

				if ((jdx)%change_rule_period==0):
					newDim = torch.randint(low=0, high=num_digit, size=(batch_size,)).to(device);

				# RNN works with size seq len X batch size X input size, in this case # chunks X 1 X pattern size + |A| + 1
				patterns = torch.cat(
										(img, torch.zeros((image_size, batch_size, num_classes+1), device=device)), dim=2
									);
				# feedback from previous trial, 1 X batch size X [0]*pattern_size + previous action + previous reward
				feedback = torch.cat(
										(torch.zeros((batch_size, image_size), device=device), 5*action.detach(), 5*reward.detach()), dim=1
									).reshape(1, batch_size, image_size+num_classes+1);

				total_input = torch.cat(
										(feedback, patterns), dim=0
									);

				# one iter of network, notice that the reward is from the previous time step
				new_v, new_h, new_dU, new_trace, (_, log_probs, value) = model.eval().forward(\
															  x = total_input.to(device),\
															  h = new_h, \
															  v = new_v, \
															  dU = new_dU, \
															  trace = new_trace);

				# sample an action
				m = torch.distributions.Categorical(logits = log_probs[-1]);
				action_idx = m.sample();
				action = torch.zeros(batch_size, num_classes, dtype=torch.float, device=device);
				action[torch.arange(batch_size), action_idx] = 1.0;

				# get reward
				reward = 2*torch.as_tensor((((label//torch.pow(10, newDim)%10)%num_classes).flatten()==action_idx), dtype=torch.float, device=device).reshape(batch_size,-1).detach()-1;

				cumReward.append(1-(reward.detach()+1)/2);

				# dUs[-1].append(new_dU[0]);
				# vs[-1].append(new_v[0]);
				# dims[-1].append(newDim);

	torch.save({
	  'model_state_dict': model.state_dict(),
	  'optimizer_state_dict': optimizer.state_dict(),
	  }, 'model_WCST');
	print(torch.mean(torch.stack(cumReward)));
			# plt.imshow(cumReward);
			# plt.show();



	
# 		plt.plot(np.sum(np.array(cumReward[-drawInt:-1]), axis=1));
# 		plt.draw()
# 		plt.pause(0.01)
# 		plt.clf()
