import torch
from modulated_full import SGRU
import torch.optim as optim
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb
from tqdm import tqdm

torch.manual_seed(0)

def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if ("ln" in name or "encoder" in name or "weight" not in name):
			no_decay.append(param);
		else:
			decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 1e-7}];

hidden_size = 32;
embedding_dim = 100;
token_num = 26;
output_num = 10;

model = SGRU(in_type = "categorical",\
			 out_type = "categorical",\
			 num_token = token_num+output_num+1,\
			 input_dim = embedding_dim,\
			 hidden_dim = hidden_size,\
			 out_dim = output_num,\
			 num_layers = 1,\
			 activation="relu",\
			 mod_rank = 16,\
       		 tie_weight= False\
			 );

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	model.to(device);
else:
	device = torch.device("cpu");

param_groups = add_weight_decay(model);

criterion = torch.nn.NLLLoss();

# optimizer = optim.SGD(param_groups, lr=1e-1, momentum=0.9, nesterov=True);
# optimizer = Lamb(param_groups, lr=4e-3);
optimizer = optim.AdamW(param_groups, lr=1e-3);
loss = 0;

n_epochs = 50;
num_pairs = 15;
train_iter = 100000;
val_iter = 10000;
test_iter = 100;
batch_size = 100;

data = [];
key_ind = range(token_num);

# scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5);
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50);
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=5e-3, factor=0.5);

# generate keys via argsort to get randomized order
keys = torch.rand(token_num, (train_iter+val_iter+test_iter)).argsort(axis=0)[:num_pairs, :]+output_num;
# generate values: values don't need to be unique
values = torch.randint(0, output_num, size=keys.shape);
# interleave the two to get data
data = torch.zeros((num_pairs*2, (train_iter+val_iter+test_iter)));
data[::2, :] = keys;
data[1::2, :] = values;
# add the start retrieval cue
data = torch.cat((data, torch.ones(1,(train_iter+val_iter+test_iter))*(output_num+token_num)), 0);
# get the query and target
query_indices = torch.randint(0, num_pairs, size=(train_iter+val_iter+test_iter,));
queries = data[query_indices*2, torch.arange((train_iter+val_iter+test_iter))];

target = data[query_indices*2+1, torch.arange((train_iter+val_iter+test_iter))].long().to(device);
data = torch.cat((data, queries.unsqueeze(0), torch.ones(1,(train_iter+val_iter+test_iter))*(output_num+token_num)), dim=0).long().to(device);

dUs = [];
val_error = [];



'''''''''''''''''''''''''''''

training

'''''''''''''''''''''''''''''

try:
	state_dict = torch.load("model_ART.dms", map_location=lambda storage, loc: storage);
	print(model.load_state_dict(state_dict["model_state_dict"]));#print(model.state_dict());
	optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
	val_error = state_dict["val_error"];
	print("model loaded successfully");
except:
	print("model failed to load");

for i in range(n_epochs):
	loss = 0;

	new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size, device);
	order = np.random.choice(np.arange(0, train_iter), replace=False, size=(train_iter));

	dUs = [];

	for idx in tqdm(range(0, train_iter, batch_size), position=0):
		new_v, new_h, new_dU, new_trace, (_, output) = model.forward(data[:,order[idx:idx+batch_size]], new_h, new_v, new_dU, new_trace);
		# dUs.append(new_dU[0]);
		loss += criterion(output[-1], target[order[idx:idx+batch_size]]);

		if ((idx+batch_size)%5000==0):
			print(loss);

		loss.backward();
# 		model.scale_grad();
		torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10);

		optimizer.step();
		scheduler.step();
		optimizer.zero_grad();

		new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size, device);

		loss = 0;

	with torch.no_grad():
		error = 0;
		order = np.random.choice(np.arange(train_iter, train_iter+val_iter), replace=False, size=(val_iter));
		new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size, device);
		for idx in range(0, val_iter, batch_size):
			new_v, new_h, new_dU, new_trace, (_, output) = model.forward(data[:,order[idx:idx+batch_size]], new_h, new_v, new_dU, new_trace);
			dUs.append(new_dU[0]);
			error += torch.sum(torch.argmax(output[-1,:,:], dim=1)!=target[order[idx:idx+batch_size]]);
			new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size, device);

	print(i, error*1.0/val_iter);
	val_error.append(error*1.0/val_iter);
	scheduler1.step(val_error[-1]);
	torch.save({
		  'model_state_dict': model.state_dict(),
		  'optimizer_state_dict': optimizer.state_dict(),
      'val_error': val_error
		  }, 'model_ART');


with torch.no_grad():
	dUs = [];
	vs = [];
	error = 0;
	order = np.random.choice(np.arange(train_iter+val_iter, train_iter+val_iter+test_iter), replace=False, size=(test_iter));
	new_h, new_v, new_dU, new_trace = model.get_init_states(1, device);
	for idx in tqdm(range(0, test_iter)):
		dUs.append([]);
		vs.append([]);
# 		errors.append(0);
		for jdx in range(0, data.shape[0]):
			new_v, new_h, new_dU, new_trace, (_, output) = model.forward(x=data[jdx:jdx+1,order[idx:idx+1]],
																h=new_h,
																v=new_v,
																dU=new_dU,
																trace=new_trace);
			dUs[-1].append(new_dU[0].squeeze());
			vs[-1].append(new_v[0].squeeze());
		error += torch.argmax(output[-1,:,:], dim=1)!=target[order[idx:idx+1]];
		new_h, new_v, new_dU, new_trace = model.get_init_states(1, device);
print(error*1.0/test_iter)
# torch.save({
# 		  'model_state_dict': model.state_dict(),
# 		  'optimizer_state_dict': optimizer.state_dict(),
# 		  }, 'model_ART');