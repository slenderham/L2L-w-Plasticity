from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ExpSineSquared
import numpy as np
import torch
from modulated_ctRNN import SGRU as RNN
import torch.optim as optim
from matplotlib import pyplot as plt
from lamb import Lamb
from tqdm import tqdm

def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if ("ln" in name or "encoder" in name or "weight" not in name):
			no_decay.append(param);
		else:
			decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-6}];

# sample gaussian process with endpoints fixed at 0
# and exp transform, since we are using strictly (or mostly) positive activation

def make_signal(batch_size, x_range):
	
	global GPs
	instr_single = x_range;
	instr = torch.as_tensor(np.tile(instr_single, (1, batch_size, 1))).transpose(0, 2);

	GP = np.random.choice(GPs);
	# sample dimensions: seq len X batch size X input dimension
	# discard the last element to avoid presenting endpoints twice

	s = GP.sample_y(instr_single.reshape(-1,1), n_samples=batch_size, random_state=np.random.randint(0, np.iinfo(np.int32).max)).transpose(0, 2, 1);

	return np.roll(s, axis=0, shift=1)/3, s/3;

batch_size = 16;
len_seq = 1000;
val_period = 10;
x_range = np.linspace(0, 1, len_seq);
train_epochs = 10000;
hidden_size = 96;

# make GP's
task_params = [(1, 0.1), (0.5, 0.25), (0.25, 0.5), (0.1, 0.5), (0.05, 1.0)];

GPs = [];

for p in task_params:
	K = ExpSineSquared(length_scale=p[1], periodicity=p[0]);
	GP = GaussianProcessRegressor(kernel=K, optimizer=None);
	GP.fit([[x_range[0]], [x_range[-1]]], [[0], [0]]);
	GPs.append(GP);

# make model
model = RNN(in_type = "continuous",\
			 out_type = "continuous",\
			 num_token = 0,\
			 input_dim = 1,\
			 hidden_dim = hidden_size,\
			 out_dim = 1,\
			 num_layers = 1,\
			 activation="relu",\
			 mod_rank = hidden_size//4,\
			 );

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device);
else:
	device = torch.device("cpu");


param_groups = add_weight_decay(model);

# optimizer = optim.SGD(param_groups, lr=1e-3, momentum=0.9);
# optimizer = LARC(raw_optimizer);

# optimizer = Lamb(param_groups, lr=1e-3);
optimizer = optim.AdamW(param_groups, lr=1e-3);

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8);
# scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2);
criterion = torch.nn.MSELoss(reduction="sum");
loss = 0;


try:
	state_dict = torch.load("model_GP");
	model.load_state_dict(state_dict["model_state_dict"]);#print(model.state_dict());
	optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
	print("model loaded successfully");
except:
	print("model failed to load");

for i in tqdm(range(train_epochs), position=0):
	loss = 0;
	new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);

	instr, target = make_signal(batch_size, x_range)
	target = torch.as_tensor(target, dtype=torch.float, device=device);
	instr = torch.as_tensor(instr, dtype=torch.float, device=device);

	output = torch.zeros(1, batch_size, 1);

	outputs = [];
	for j in range(len_seq):
		new_v, new_h, new_dU, new_trace, (last_layer, output) = model.forward(instr[j:j+1]-output.detach(), new_h, new_v, new_dU, new_trace);
		loss += criterion(output.squeeze(), target[j].squeeze());
		# loss += criterion(output.view(output.size(0)*output.size(1), output.size(2)), target.squeeze().view(target.shape[0]*target.shape[1], target.shape[2]));
	# dUNorm = (torch.relu(model.rnns[0].alpha)*new_dU[0]).pow(2).sum();
		outputs.append(output);

	loss.backward();
# 	model.scale_grad();

	torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10);
	optimizer.step();
	optimizer.zero_grad()
	scheduler.step();

	if (i)%val_period==0:
		torch.save({
		  'model_state_dict': model.state_dict(),
		  'optimizer_state_dict': optimizer.state_dict(),
		  }, 'model_GP');torch.save({'output': torch.stack(outputs), 'target': target}, 'out_and_target');
		print(loss/batch_size);

