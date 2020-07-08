import torch
import os, sys
from modulated_full import SGRU as RNN
import torch.optim as optim
from lamb import Lamb
import numpy as np
from matplotlib import pyplot as plt
from torchtext import data, vocab
from torchtext.datasets import PennTreebank
from tqdm import tqdm
from dataloader import batchify, get_batch, Corpus


torch.manual_seed(0);

def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if ("ln" in name or "encoder" in name or "weight" not in name):
			no_decay.append(param);
		else:
			decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-6}];

n_epochs = 30;
bptt_len = 200;
batch_size = 20;
dataset = 'L2L-w-STDP/data/pennchar/'

if torch.cuda.is_available():
	device = torch.device("cuda:0");
else:
	device = torch.device("cpu");

# train_iter, val_iter, test_iter = PennTreebank.iters(batch_size=batch_size, bptt_len=bptt_len, device = device);
import os
import hashlib
f = 'corpus.{}.data'.format(hashlib.md5(dataset.encode()).hexdigest())
if os.path.exists(f):
	print('Loading cached dataset...')
	corpus = torch.load(f)
else:
	print('Producing dataset...')
	corpus = Corpus(dataset)
	torch.save(corpus, f)

train_data = batchify(corpus.train, batch_size, device)
val_data = batchify(corpus.valid, batch_size, device)
test_data = batchify(corpus.test, 1, device)

print(len(corpus.dictionary))


model = RNN(in_type = "categorical",\
			 out_type = "categorical",\
			 num_token = len(corpus.dictionary),\
			 input_dim = 256,\
			 hidden_dim = 256,\
			 out_dim = len(corpus.dictionary),\
			 num_layers = 2,\
			 activation="softplus",\
			 mod_rank = 128,\
			 dropout_e = 0.0, dropout_i = 0.1, dropout_h = 0.1, dropout_o = 0.1, dropout_w = 0.15,
			 );

model.to(device);

param_groups = add_weight_decay(model);

# optimizer = Lamb(param_groups, lr=1e-2, min_trust=0.1);
optimizer = optim.AdamW(param_groups, lr=5e-4);
# optimizer = optim.SGD(param_groups, lr=1e-4);
#schedulers = [
#				optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.6),
#				optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_iter)//4)
#			 ];

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1);

criterion = torch.nn.NLLLoss();
loss = 0;

#new_vs = torch.zeros(512, n_epochs*len_seq);

devloss = [];

try:
  state_dict = torch.load("model_PTB.dms");
  def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
      if name not in own_state:
        print(name);continue
      if isinstance(param, torch.nn.Parameter):
        # backwards compatibility for serialized parameters
        param = param.data
      own_state[name].copy_(param)
  load_my_state_dict(model, state_dict["model_state_dict"]);#print(model.state_dict());
  optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
  devloss = state_dict['loss'];
except:
	print("Unexpected error:", sys.exc_info()[0]);

for i in range(n_epochs):
	print("epoch ", i);
	loss = 0;
	act_loss = 0;
	new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
	model.train();
	for batch_num in tqdm(range(0, train_data.size(0)-1, bptt_len), position=0):
		text, target = get_batch(train_data, batch_num, bptt_len);

		new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.forward(x = text, \
														  v = new_v, \
														  h = new_h, \
														  dU = new_dU, \
														  trace = new_trace);

		loss += criterion(output.view(output.size(0)*output.size(1), output.size(2)), target);

		if (batch_num%5000==0):
			print(loss);
		loss.backward();
		loss = 0;

		if (batch_num+1)%1==0:
			# model.scale_grad();
			torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0);

			optimizer.step();
			optimizer.zero_grad();

		model.detach(new_v, new_h, new_dU, new_trace);

	with torch.no_grad():
		model.eval();
		for batch_num in tqdm(range(0, val_data.size(0)-1, bptt_len)):
			text, target = get_batch(val_data, batch_num, bptt_len);

			new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.forward(x = text, \
															  v = new_v, \
															  h = new_h, \
															  dU = new_dU, \
															  trace = new_trace);
			loss += criterion(output.view(output.size(0)*output.size(1), output.size(2)), target)/(val_data.size(0)//bptt_len);

		model.train();
		devloss.append(loss);

	torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': devloss,
			}, 'model_PTB');

	print(loss);
	scheduler.step();
	loss = 0;

with torch.no_grad():
	model.eval();
	loss = 0;
	new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
	for batch_num in tqdm(range(0, test_data.size(0)-1, bptt_len)):
		text, target = get_batch(test_data, batch_num, bptt_len);
		new_v, new_h, new_dU, new_trace, (_, output) = model.forward(x = text, \
													  v = new_v, \
													  h = new_h, \
													  dU = new_dU, \
													  trace = new_trace);

		loss += criterion(output.view(output.size(0)*output.size(1), output.size(2)), target);
	print(loss/(test_data.size(0)//bptt_len));


