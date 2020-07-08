import torch
from model.modulated_AC import SGRU
import torch.optim as optim
import gym
import numpy as np
from matplotlib import pyplot as plt
from utils.lamb import Lamb
from utils.LARC import LARC
from utils.a2c import compute_loss
'''
	a wisconsin card sort task with no explicit task instruction
	the input are binary patterns, four chunks of four one hot vectors
	the task is to output the identity of one of the chunks
	reward given as +-1
	trained with NLL
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

env = gym.make('CartPole-v1');
# env.seed(1);

a_space_shape = env.action_space.n;
o_space_shape = env.observation_space.shape[0];



n_epochs = 500;
beta_v = 0.8;
beta_a = 0.05;

cumReward = []

for k in range(1):
	model = SGRU(in_type = "continuous",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = o_space_shape+a_space_shape+1,\
			 hidden_dim = 256,\
			 out_dim = a_space_shape,\
			 num_layers = 2,\
			 activation="poisson",\
			 mod_rank = 32\
			 );

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		model.to(device)

	# raw_optimizer = optim.SGD(model.parameters(), lr=1);
	# optimizer = LARC(raw_optimizer);

	param_group = add_weight_decay(model);

	optimizer = Lamb(param_group, lr=1e-2);
	scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6);

	loss = 0;

	cumReward.append([]);

	for i in range(n_epochs):
		env.reset();

		dUs = [];

		# initialize loss and reward
		loss_ent = 0;
		reward = torch.zeros(1, 1);
		new_h, new_v, new_dU, new_trace = model.get_init_states();
		action = torch.zeros(1, a_space_shape, dtype=torch.float);
		action_idx = torch.randint(a_space_shape, (1,1)).data;
		action[:, action_idx] = 1.0;

		(obs, reward, done, _info) = env.step(action_idx.item());

		log_prob_of_act = [];
		values = [];
		rewards = [];

		cumReward[-1].append(0);

		if (i!=0 and i%1000==0):
			beta_a /= 5;
	# 		scheduler1.step();

		while not done:
			if i>300:
				env.render();

			total_input = torch.cat((torch.tensor(obs, dtype=torch.float).reshape(1,-1), action.detach(), torch.tensor([[reward]],dtype=torch.float).reshape(1,-1)), dim=1);

			# one iter of network, notice that the reward is from the previous time step
			new_v, new_h, new_dU, new_trace, (log_probs, value) = model.forward(\
														  x = total_input.reshape(1,-1),\
														  h = new_h, \
														  v = new_v, \
														  dU = new_dU, \
														  trace = new_trace);
			# sample an action
			m = torch.distributions.Categorical(logits = log_probs);
			action_idx = m.sample();
			action = torch.zeros(1, a_space_shape, dtype=torch.float);
			action[:, action_idx] = 1.0;

			# calculate entropy loss
			loss_ent += torch.sum(log_probs*torch.exp(log_probs))

			# get reward
			(obs, reward, done, _info) = env.step(action_idx.item());

			# save reward, log_prob, value;
			rewards.append(reward);
			cumReward[-1][-1] += reward;
			log_prob_of_act.append(m.log_prob(action_idx));
			values.append(value);

			dUs.append(new_dU[0]);

		loss = compute_loss(rewards=torch.tensor(rewards), \
							 values=values, \
							 log_probs=log_prob_of_act, \
							 gamma=0.95, \
							 N=64, \
							 beta_v=beta_v, \
						 )\
				+ beta_a*loss_ent;

		optimizer.zero_grad();
		loss.backward();
		model.scale_grad();

		print(i, cumReward[-1][-1]);
		torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1);

		optimizer.step();

	#	with torch.no_grad():
	#		reward = torch.zeros(1, 1);
	#		new_h, new_v, new_dU, new_trace = model.get_init_states();
	#		output = torch.zeros(1, bits);
	#		countDown = 0;
	##		model.detach(new_v, new_h, new_dU, new_trace);
	#
	#		for s in range(val_len):
	#			instrInts, instrPattern, newDim, countDown = sampler(prevDim, countDown);
	#
	##			if i<300:
	#			reward = torch.tensor(newDim-1, dtype=torch.float).reshape(1,1);
	##
	#			total_input = torch.cat((instrPattern.reshape(1,-1), action.detach(), reward.detach()), dim=1);
	#
	#			# one iter of network, notice that the reward is from the previous time step
	#			new_v, new_h, new_dU, new_trace, (log_probs, value) = model.forward(\
	#															x = total_input.reshape(1,-1),\
	#															v = new_v,\
	#															h = new_h,\
	#															dU = new_dU,\
	#															trace = new_trace);
	#
	#
	#			# sample an action
	#
	#			m = torch.distributions.Categorical(logits = log_probs);
	#			action_idx = m.sample();
	#			action = torch.zeros(1, bits);
	#			action[:, action_idx] = 1;
	#
	#			reward = torch.tensor((instrInts[newDim]==action_idx), dtype=torch.float).reshape(1,1);
	##			print("Inputs: ", instrInts, "; Pay attention to: ", prevDim, "; You chose: ", action, " ", output);


	#
	#	if (i!=0 and i%drawInt==0):
	#		print("epoch ", i);
	#		plt.plot(accTrain[:i]/len_seq);
	#		plt.plot(accVal[:i]/val_len);
	#		plt.draw()
	#		plt.pause(0.0001)
	#		plt.clf()
	##		scheduler1.step();
	##	optimizer.zero_grad();
	#	print(accTrain[i], accVal[i]);
	#

