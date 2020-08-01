import torch
from modulated_ctRNN_AC import SGRU
import torch.optim as optim
from scipy.stats import ortho_group
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb
from a2c import compute_loss
import tqdm

'''
	simpler WCST
	the episodes are predetermined, but whether the dimension to attend to changes from trial to trial
	however, it stays constant during each trial
	the task is to quickly

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

	randInts = torch.tensor(np.random.randint(low = 0, high = val, size = (chunks,)));
	pattern = data[randInts, :].flatten();

	return randInts, pattern;


# how many dimensions
chunks = 4;

# the size of each input dimension
bits = 4;

# the possible values each input dimension can take
val = 4;


sampler = lambda : sample_card(chunks, bits, val);

model = SGRU(in_type = "binary",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = bits*chunks+val+1,\
			 hidden_dim = 32,\
			 out_dim = val,\
			 num_layers = 1,\
			 activation="relu",\
			 mod_rank = 8\
			 );

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device);
else:
	device = torch.device("cpu");

param_groups = add_weight_decay(model);

optimizer = Lamb(param_groups, lr=1e-2);
# optimizer = optim.AdamW(param_groups, lr=4e-3);
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6);

loss = 0;

n_epochs = 1500;
len_seq = 100;
val_len = 100;
num_samples = 30;
turns = 10;

beta_v = 0.1;
beta_a = 1;

drawInt = 100;


data = torch.tensor(ortho_group.rvs(bits), dtype=torch.float);


for i in tqdm.tqdm(range(n_epochs), position=0, leave=True):

	data = torch.tensor(ortho_group.rvs(bits), dtype=torch.float);

	# initialize loss and reward
	loss_ent = 0;
	reward = torch.zeros(1, 1);
	new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
	action = torch.eye(val)[torch.randint(val, (1,)),:];

	log_prob_of_act = [];
	values = [];
	rewards = [];

	for idx in range(len_seq):
		if (idx%turns==0):
			instrInts, instrPattern = sampler();
			newDim = np.random.randint(low=0, high=chunks, size=1);

		# RNN works with size seq len X batch size X input size, add the first dimension of sequence of length 1
		total_input = torch.cat((instrPattern.reshape(1, -1)+torch.randn(instrPattern.shape)/bits, chunks*action.detach(), chunks*reward.detach()), dim=1);
		# one iter of network, notice that the reward is from the previous time step
		new_v, new_h, new_dU, new_trace, (log_probs, value) = model.forward(\
													  x = total_input,\
													  h = new_h, \
													  v = new_v, \
													  dU = new_dU, \
													  trace = new_trace);


		# sample an action
		m = torch.distributions.Categorical(logits = log_probs);
		action_idx = m.sample();
		action = torch.zeros(1, val, dtype=torch.float);
		action[:, action_idx] = 1.0;

		# calculate entropy loss
		loss_ent += torch.sum(log_probs*torch.exp(log_probs))

		# get reward
		reward = torch.tensor((instrInts[newDim].flatten()==action_idx), dtype=torch.float).reshape(1,-1).detach();

		# save reward, log_prob, value;
		rewards.append(reward);
		log_prob_of_act.append(m.log_prob(action_idx));
		values.append(value);

	loss = compute_loss(rewards=torch.tensor(rewards), \
						 values=values, \
						 log_probs=log_prob_of_act, \
						 gamma=0.95, \
						 N=25, \
						 beta_v=beta_v, \
					 )\
			+ beta_a*loss_ent;

	loss.backward();
# 	model.scale_grad();

#	print(loss);
	torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 5);
	optimizer.step();
	optimizer.zero_grad();


	if ((i+1)%50==0):
		dUs = [];
		vs = [];
		dims = [];
		cumReward = [];

		with torch.no_grad():
			for j in range(num_samples):

				dUs.append([]);
				vs.append([]);
				dims.append([]);
				cumReward.append([]);


				data = torch.tensor(ortho_group.rvs(bits), dtype=torch.float);
				newDims = np.random.choice(range(chunks), size=chunks, replace=False);

				reward = torch.zeros(1, 1);
				new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
				action = torch.eye(val)[torch.randint(val, (1,)),:];

				for idx in range(val_len):
					if (idx%turns==0):
						instrInts, instrPattern = sampler();
						newDim = newDims[(idx//10)%chunks];

					# RNN works with size seq len X batch size X input size, add the first dimension of sequence of length 1
					total_input = torch.cat((instrPattern.reshape(1, -1)+torch.randn(instrPattern.shape)/bits, chunks*action.detach(), chunks*reward.detach()), dim=1);
					# one iter of network, notice that the reward is from the previous time step
					new_v, new_h, new_dU, new_trace, (log_probs, value) = model.forward(\
																  x = total_input,\
																  h = new_h, \
																  v = new_v, \
																  dU = new_dU, \
																  trace = new_trace);


					# sample an action
					m = torch.distributions.Categorical(logits = log_probs);
					action_idx = m.sample();
					action = torch.zeros(1, val, dtype=torch.float);
					action[:, action_idx] = 1.0;

					reward = torch.tensor((instrInts[newDim].flatten()==action_idx), dtype=torch.float).reshape(1,-1).detach();
					cumReward[-1].append(1-reward.item());

					dUs[-1].append(new_dU[0]);
					vs[-1].append(new_v[0]);
					dims[-1].append(newDim);

			print(np.mean(cumReward));

	if (i!=0 and i%drawInt==0):
		beta_a = max(beta_a*0.5, 0.001);
		scheduler1.step();
# 		plt.plot(np.sum(np.array(cumReward[-drawInt:-1]), axis=1));
# 		plt.draw()
# 		plt.pause(0.01)
# 		plt.clf()

# torch.save({'vs': vs, 'dUs': dUs, 'model': model, 'cumReward': cumReward, 'dims': dims}, 'WCST_STDP');
