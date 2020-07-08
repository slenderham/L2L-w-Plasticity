import torch
from model.modulated_full import SGRU as RNN
from utils.lamb import Lamb
from utils.LARC import LARC
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder

'''
	using Andrej Karpathy's tiny shakespeare dataset
'''

# torch.manual_seed(0);

def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if ("ln" in name or "encoder" in name or "weight" not in name):
			no_decay.append(param);
		else:
			decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-5}];


data = np.array(list(open("../data/sonnets.txt", "r").read().lower())).ravel();
enc = LabelEncoder();
data = torch.tensor(enc.fit_transform(data), dtype=torch.long);

print((enc.classes_));

hidden_size = 256;

model = RNN(in_type = "categorical",\
			 out_type = "categorical",\
			 num_token = len(enc.classes_),\
			 input_dim = hidden_size,\
			 hidden_dim = hidden_size,\
			 out_dim = len(enc.classes_),\
			 num_layers = 1,\
			 activation="swish",\
			 mod_rank = 32,\
			 dropout_e = 0.1, dropout_i = 0.1, dropout_h = 0.1, dropout_o = 0.1\
			 );

if torch.cuda.is_available():
	device = torch.device("cuda");
	model = model.to(device)
else:
  device = torch.device("cpu");

param_groups = add_weight_decay(model);

# raw_optimizer = optim.SGD(param_groups, lr=1, momentum=0.9);
# optimizer = LARC(raw_optimizer);

optimizer = Lamb(param_groups, lr=1e-3, min_trust=0.25);

scheduler = [
		optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8),\
#		optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
			];
criterion = torch.nn.NLLLoss();
loss = 0;

n_epochs = 10;
len_seq = data.shape[0];
sample_len = 200;
sample_time = 1000;
cutoff = 60;

#new_vs = torch.zeros(512, n_epochs*len_seq);

for i in range(n_epochs):
	print("epoch ", i);
	loss = 0;
	new_h, new_v, new_dU, new_trace = model.get_init_states();

	for idx in range(len_seq-1):

		new_v, new_h, new_dU, new_trace, output = model.forward(x = data[idx].view(1,1).to(device), \
														  v = new_v, \
														  h = new_h, \
														  dU = new_dU, \
														  trace = new_trace);

		loss += criterion(output, data[idx+1].view(1).to(device));

		if (idx%cutoff==0 and idx!=0):
			loss.backward(retain_graph=True);

			print(idx, loss/cutoff);
			loss = 0;
			model.scale_grad();
			torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1);
			optimizer.step();
			optimizer.zero_grad();
			for s in scheduler:
				s.step();

			model.detach(new_h, new_v, new_dU, new_trace);


		if (idx%sample_time==0):
			model.eval();
			with torch.no_grad():
				for s in range(sample_len):
					randChar = torch.multinomial(torch.exp(output).squeeze(),1);
					print(enc.inverse_transform(randChar.cpu())[0],end="");
					new_v, new_h, new_dU, new_trace, output = model.forward(x = randChar.view(1,1), \
															 v = new_v, \
															 h = new_h, \
															 dU = new_dU, \
															 trace = new_trace);
				print();
				print("-----------------");

			model.train();

	print(loss);
	loss.backward();
	torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 5);
	optimizer.step();
	optimizer.zero_grad()
	for s in scheduler:
		s.step();

