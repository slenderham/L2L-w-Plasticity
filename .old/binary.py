import torch
from model.gated import SGRU
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms



model = SGRU("continuous", "continuous", 0, 16, 32, 16);
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.NLLLoss(reduction="sum");
loss = 0;

n_epochs = 100;
len_seq = 1200;

new_vs = torch.zeros(16, n_epochs*len_seq);

target = torch.randint(0, 2, (16, len_seq), dtype=torch.float);
for i in range(n_epochs):
	loss = 0;
	new_v, new_h, new_dU, new_trace = model.rnn.get_init_states();
	for idx in range(len_seq):
		new_v, new_h, new_dU, new_trace, output = model.forward(target[:,idx].view(-1, 1), new_v, new_h, new_dU, new_trace);
		new_vs[:,i*len_seq+idx] = output;
		loss += criterion(output, target[:,idx+1].view(1, -1).type(torch.long));
	print(loss);
	loss.backward();
	torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1e2)
	optimizer.step();