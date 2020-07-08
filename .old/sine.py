import torch
from modulated_full import SGRU as RNN
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


# this is not working, try with continuos time RNN instead of GRU

torch.manual_seed(0)

def model_save(fn):
	with open(fn, 'wb') as f:
		torch.save([model, criterion, optimizer], f)

def model_load(fn):
	global model, criterion, optimizer
	with open(fn, 'rb') as f:
		model, criterion, optimizer = torch.load(f)

def sum_of_sine(num_step, dt, pre = [1, 1.5, 2, 3, 4, 6]):
	x = torch.linspace(0, num_step*dt, num_step);
	y = torch.zeros(1, num_step);
	choose = torch.randint(len(pre), (2,1));
	y += torch.sin(x*np.pi/pre[choose[0]]) + torch.sin(x*np.pi/pre[choose[1]]);
	return torch.sin(x*np.pi/pre[0]), y/2;

hidden_size = 64;

model = RNN(in_type = "continuous",\
			 out_type = "continuous",\
			 num_token = 0,\
			 input_dim = 1,\
			 hidden_dim = hidden_size,\
			 out_dim = 1,\
			 activation="swish",\
			 num_layers=1,\
			 mod_rank=1
			 );


if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device)
else:
	device = torch.device("cpu");

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95);
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2);
criterion = torch.nn.MSELoss(reduction="mean");
loss = 0;

n_epochs = 600;
len_seq = 120;
val_len = 300;

dUs = torch.zeros(val_len+len_seq, hidden_size, hidden_size);
outputs = torch.zeros(1, val_len);

instr, target = sum_of_sine(val_len, 0.1);

print(target.shape);

loss_np = [];

for i in range(n_epochs):
	loss = 0;
	new_h, new_v, new_dU, new_trace =  model.get_init_states(batch_size=1, device=device);
	output = torch.zeros(1,1,1);
	instr, target = sum_of_sine(val_len, 0.1);

	for idx in range(len_seq//5):
		new_v, new_h, new_dU, new_trace, output = model.forward(target[:,idx].reshape(1,1,-1), new_v, new_h, new_dU, new_trace);
		dUs[idx, :,:] = new_dU[0];
		loss += criterion(output, target[:,idx+1].view(1, -1));

	for idx in range(len_seq//5, len_seq-1):
		new_v, new_h, new_dU, new_trace, output = model.forward(output.reshape(1,1,1), new_v, new_h, new_dU, new_trace);
		dUs[idx, :,:] = new_dU[0];
		loss += criterion(output, target[:,idx+1].view(1, -1));

	loss.backward();
#	model.scale_grad();

#	 adjust the scale of the gradient - some terms appear in many places
#	model.rnn.alpha.grad /= hidden_size;

	torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 5);
	optimizer.step();
	optimizer.zero_grad()
	scheduler.step();

	loss = 0;
#	new_v[:] = [v.detach() for v in new_v];
#	new_h[:] = [h.detach() for h in new_h];
#	new_dU[:] = [dU.detach() for dU in new_dU];
#	new_trace[:] = [trace.detach() for trace in new_trace];

	with torch.no_grad():
		new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
		output = torch.zeros(1,1);
		instr, target = sum_of_sine(val_len, 0.1);

		for idx in range(val_len):
			new_v, new_h, new_dU, new_trace, output = model.forward(output.reshape(1,1,-1), new_v, new_h, new_dU, new_trace);
			outputs[:, idx] = output.squeeze();
			dUs[len_seq+idx,:,:] = new_dU[0];
			loss += criterion(output, target[:,idx].view(1, -1));

	print(i, loss);
	loss_np.append(loss);
