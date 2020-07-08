import torch


num_healthy_samples = 30
num_ablated_trials = 30;

model = torch.load('model_WCST')['model'];

# first simulate a set of trajectories
dUs = [];
vs = [];
dims = [];
cumReward = [];

dUsA = [];
vsA = [];
dimsA = [];
cumRewardA = [];

with torch.no_grad():
	for j in range(num_healthy_samples):

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

	print(np.mean(cumReward[-1]));

	for j in range(num_ablated_trials):

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

	print(np.mean(cumReward[-1]));


