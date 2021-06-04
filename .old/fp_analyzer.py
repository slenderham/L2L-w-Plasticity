"""class for fixed point analysis"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from modulated_AC import SGRU
import numpy as np
from random import randint
from matplotlib import pyplot as plt
import itertools

torch.manual_seed(37);

class FixedPoint(object):
	def __init__(self, model, device, gamma=0.01, speed_tor=1e-06, max_epochs=200000,
				 lr_decay_epoch=10000):
		self.model = model
		self.device = device
		self.gamma = gamma
		self.speed_tor = speed_tor
		self.max_epochs = max_epochs
		self.lr_decay_epoch = lr_decay_epoch

		self.model.eval()

	def calc_speed(self, v, dU, const_signal):
		new_v, _, _, _, _ = self.model.forward(x = input_signal, \
													v = v, \
													h = torch.relu(v), \
													dU = dU, \
													trace = torch.zeros_like(dU),
													turn_off_plasticity=True);
		speed = torch.norm(new_v-v);

		return speed

	def find_fixed_point(self, init_v, const_signal, const_fast_weight, view=False):
		v_result = init_v.clone();
		gamma = self.gamma
		result_ok = True
		i = 0
		pbar = tqdm(total=self.max_epochs);
		while True:
			pbar.update(1);

			# make a copy of the previous guess
			v_guess = torch.as_tensor(v_result, device=self.device);
			# run the model and calculate approximate derivative
			speed = self.calc_speed(v_guess, const_fast_weightm, const_signal);

			if view and i % 1000 == 0:
				print(f'epoch: {i}, speed={speed.item()}')
			if speed.item() < self.speed_tor:
				print(f'epoch: {i}, speed={speed.item()}')
				break

			# backprop to the hidden state
			speed.backward();

			if i % self.lr_decay_epoch == 0 and i > 0:
				gamma *= 0.5
			if i == self.max_epochs:
				# failed to converge
				print(f'forcibly finished. speed={speed.item()}')
				result_ok = False
				break
			
			i += 1

			# we only care about the cell state: everything else should be kept constant
			v_result = v_guess - gamma * v_guess.grad

		fixed_point = v_result
		return fixed_point, result_ok

	def calc_jacobian(self, fixed_point, const_signal_tensor, const_fast_weight):
		fixed_point = torch.unsqueeze(fixed_point, dim=1)
		fixed_point = Variable(fixed_point).to(self.device)
		fixed_point.requires_grad = True

		new_v, _, _, _, _ = self.model.forward(x = input_signal, \
													v = fixed_point, \
													h = torch.relu(fixed_point), \
													dU = const_fast_weight, \
													trace = torch.zeros_like(const_fast_weight),
													turn_off_plasticity=True);


		jacobian = torch.zeros(self.model.hidden_dim, self.model.hidden_dim);
		for i in range(self.model.n_hid):
			output = torch.eye(self.model.n_hid)[i].to(self.device)
			jacobian[:, i] = torch.autograd.grad(y, x, grad_outputs=output, retain_graph=True)[0]

		jacobian = jacobian.numpy().T

		return jacobian

if __name__=="__main__":
	state_dict = torch.load('./model_WCST');
	model = SGRU(in_type = "continuous",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = bits+val+1,\
			 hidden_dim = 32,\
			 out_dim = val,\
			 num_layers = 1,\
			 activation="relu",\
			 mod_rank = 16\
			);
	model.load_state_dict(state_dict['model_state_dict']);
	FP = FixedPoint(model=model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

	def sample_card(chunks, bits, val):
		'''
			input:
				chunks: the number of dimensions (color, shape, numer of shapes...)
				bits: the number of variations along each dimensions
			output:
				randInts: the "card" with different feature along each dimension
				pattern: binary coded
		'''
		global data;

		randInts = torch.tensor(np.random.randint(0, val, size=(chunks,)));
		pattern = data[randInts, :];

		return randInts.to(device), pattern.to(device);

	# how many dimensions
	chunks = 3;

	# the possible values each input dimension can take
	val = 4;

	# the size of each input dimension
	bits = 4;

	# trials per rule
	tpr = 20;

	sampler = lambda : sample_card(chunks, bits, val);

	model = SGRU(in_type = "continuous",\
				 out_type = "categorical",\
				 num_token = 0,\
				 input_dim = bits+val+1,\
				 hidden_dim = 32,\
				 out_dim = val,\
				 num_layers = 1,\
				 activation="relu",\
				 mod_rank = 16\
				);
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		model.to(device);
	else:
		device = torch.device("cpu");

	len_seq = 80;

	# generate all rule combination, each for two times
	newDims = list(itertools.product(range(chunks), repeat=len_seq//tpr))*2;

	# data = torch.tensor(ortho_group.rvs(bits), dtype=torch.float, device=device);
	data = torch.eye(bits);

	# load the model
	state_dict = torch.load("model_WCST");
	print(model.load_state_dict(state_dict["model_state_dict"]));
	training_curve = state_dict["cumReward"];

	# run the model for samples
	hiddens = [];
	inputs = [];
	dims = [];

	with torch.no_grad():
		cumReward = [];
		# for each sample
		for j in range(num_samples):

			hiddens.append([]);
			inputs.append([]);
			dims.append([]);

			reward = torch.zeros(1, 1, device=device);
			new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
			action = torch.zeros(1, val, device=device);

			instrInts, instrPattern = sampler();
			newDim = newDims[j][0];

			next_change = tpr;

			for idx in range(len_seq):
				instrInts, instrPattern = sampler();
				if (idx==next_change):
					newDim = newDims[j][next_change//tpr];
					next_change += tpr;
				
				# RNN works with size seq len X batch size X input size, in this case # chunks X 1 X pattern size + |A| + 1
				patterns = torch.cat(
							(instrPattern.reshape(chunks, 1, bits), torch.zeros((chunks, 1, val+1), device=device)), dim=2
						);
				# feedback from previous trial, 1 X 1 X [0]*pattern_size + previous action + previous reward
				feedback = torch.cat(
							(torch.zeros((1, bits), device=device), action.detach(), reward.detach()), dim=1
						).reshape(1, 1, bits+val+1);

				total_input = torch.cat(
							(feedback, patterns), dim=0
						);

				# run the network and collect samples
				for jdx in range(total_input.shape[0]):
					new_v, new_h, new_dU, new_trace, (last_layer_out, log_probs, value) = model.train().forward(\
										x = total_input[jdx:jdx+1,...].to(device),\
										h = new_h, \
										v = new_v, \
										dU = new_dU, \
										trace = new_trace);
					hiddens[-1].append({'v':new_v[0], 'h':new_h[0], 'dU':new_dU[0], 'trace':new_trace[0]});
					dims[-1].append(newDim);
					inputs[-1].append(total_input[jdx:jdx+1,...]);

				# sample an action
				m = torch.distributions.Categorical(logits = log_probs[-1]);
				action_idx = m.sample();
				action = torch.zeros(1, val, dtype=torch.float, device=device);
				action[:, action_idx] = 1.0;

				# get reward
				reward = torch.as_tensor((instrInts[newDim].flatten()==action_idx), dtype=torch.float, device=device).reshape(1,-1).detach();
				cumReward[-1].append(1-reward.item());

		print(np.mean(cumReward));
		plt.imshow(cumReward);

	fps = [];
	# now for each timestep of each sample, fix input and fast weight, calculate fixed point
	for i in range(num_samples):
		fp.append([]);
		for j in range(len_seq*(chunks+1)):
			FixedPoint.find_fixed_point(init_v=hiddens[i][j]['v'], const_signal=inputs[i][j], const_fast_weight=hiddens[i][j]['dU']);
			