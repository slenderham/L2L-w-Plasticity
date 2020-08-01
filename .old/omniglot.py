import torch
from torchvision import datasets, transforms
from modulated_full import SGRU
import torch.optim as optim
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb
from tqdm import tqdm
from PIL import Image

from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine
from torchmeta.utils.data import BatchMetaDataLoader

def add_weight_decay(model):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
			if ("ln" in name or "encoder" in name or "weight" not in name):
					no_decay.append(param);
			else:
					decay.append(param);

	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 0.}];

def offset(batch):
	global batch_size, ways, shots, img_size;

	# batch X (way*shot) X 1 X img_size X img_size
	inputs, targets = batch;

	# pick the instances to show
	examples = torch.argsort(torch.rand(ways, 20), dim=1)[:,:shots]; # -> ways X shots, each row is the instances
	examples = ((torch.arange(ways)*20).reshape(ways, 1)+examples).flatten(); # -> (ways*shots)
	perm = examples[torch.randperm(ways*shots)];

	inputs, targets = inputs[:,perm], targets[:,perm];

	# training input image: reshape to (way*shot) X batch X img_size^2; add zeros to the end
	inputs = inputs.transpose(0, 1);
	inputs = (inputs-inputs.mean())/(inputs.std()+1e-8);
	# inputs = torch.cat([inputs, torch.zeros(1, batch_size, img_size**2)], dim=0);
	# training targets: make one hot vectors; add zeros to beginning
	targets = targets.transpose(0, 1);
	targets_onehot = torch.zeros(ways*shots, batch_size, ways).scatter(2, targets.unsqueeze(2), 1);
	targets_onehot = 20*torch.cat([torch.zeros(1, batch_size, ways), targets_onehot], dim=0);
	
	# total = torch.cat([inputs, targets_onehot[:-1]], dim=2);

	return [inputs.to(device), targets_onehot[:-1].to(device)], targets.to(device)

def shotAccuracy(output, target):
	acc = torch.zeros(shots);
	for i in range(batch_size):
		whichShot = torch.zeros(ways, dtype=torch.int);
		for j in range(ways*shots):
			acc[int(whichShot[target[j, i]].item())] += float(target[j, i]==output[j, i]);
			whichShot[target[j, i]] += 1;
	acc /= ways*batch_size;
	return acc;

n_epochs = 12;
batch_size = 32;
ways = 5;
shots = 10;
img_size = 28;
val_batches = 15;

AR = 1e-2;
TAR = 1e-2;

# RandomAffine(degrees=11.25, translate=(0.1, 0.1)),

train_data = Omniglot("data",
						 num_classes_per_task=ways,
						 transform=Compose([Resize(img_size, interpolation=Image.LANCZOS),  ToTensor()]),
						 target_transform=Categorical(num_classes=ways),
						 class_augmentations=[Rotation([90, 180, 270])],
						 meta_train=True,
						 download=True);
train_iter = BatchMetaDataLoader(train_data, batch_size=batch_size);

val_data = Omniglot("data",
						 num_classes_per_task=ways,
						 transform=Compose([Resize(img_size, interpolation=Image.LANCZOS), ToTensor()]),
						 target_transform=Categorical(num_classes=ways),
						 meta_val=True,
						 download=True);
val_iter = BatchMetaDataLoader(val_data, batch_size=batch_size);

test_data = Omniglot("data",
						 num_classes_per_task=ways,
						 transform=Compose([Resize(img_size, interpolation=Image.LANCZOS), ToTensor()]),
						 target_transform=Categorical(num_classes=ways),
						 meta_test=True,
						 download=True);
test_iter = BatchMetaDataLoader(test_data, batch_size=batch_size);

model = SGRU(in_type = "image++",\
			 out_type = "categorical",\
			 num_token = 0,\
			 input_dim = ways,\
			 hidden_dim = 256,\
			 out_dim = ways,\
			 num_layers = 1,\
			 activation="relu",\
             mod_rank= 128,\
			);print(model);

if torch.cuda.is_available():
		device = torch.device("cuda:0")
		model.to(device);
else:
		device = torch.device("cpu");

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=1e-3);
# scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=5e-3, factor=0.5);
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=5e3, gamma=0.5)
scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50);

criterion = torch.nn.NLLLoss();

loss = 0;
trainShotAcc = torch.zeros(shots);
val_errors = [];

try:
	state_dict = torch.load("model_omniglot");
	model.load_state_dict(state_dict["model_state_dict"]);
	optimizer.load_state_dict(state_dict["optimizer_state_dict"]);#scheduler1.step(1500);
	val_errors = state_dict['val_errors'];
	print("model loaded successfully");
except:
	print("model failed to load");

for i in range(n_epochs):
	for idx, batch in tqdm(enumerate(train_iter), position=0):
		input_total, label = offset(batch);
		new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
		new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.train().forward(\
																						  x = input_total,\
																						  h = new_h, \
																						  v = new_v, \
																						  dU = new_dU, \
																						  trace = new_trace);

		loss = criterion(output.reshape(-1, ways), label.reshape(-1));
		trainShotAcc += shotAccuracy(torch.argmax(output, dim=2), label)/50;

		# loss += AR*(last_layer_out.pow(2).mean());
		# loss += TAR*(last_layer_out[1:]-last_layer_out[:-1]).pow(2).mean();

		loss.backward(); #model.scale_grad();
		torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10.0);
		optimizer.step();
		optimizer.zero_grad();
		scheduler1.step();
		scheduler2.step();

		if (torch.isnan(loss)): exit();
		if (idx+1)%50==0:
			print(loss);
			print(trainShotAcc, torch.mean(trainShotAcc));
			trainShotAcc = torch.zeros(shots);
			valShotAccuracy = torch.zeros(shots);
			if (not torch.isnan(loss)): 
				torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_errors': val_errors}, 'model_omniglot');
			with torch.no_grad():
				error = 0;
				for jdx, batch in tqdm(enumerate(val_iter), position=0):
					if (jdx>=val_batches):
						break;
					input_total, label = offset(batch);
					new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
					new_v, new_h, new_dU, new_trace, (_, output) = model.eval().forward(\
															  x = input_total,\
															  h = new_h, \
															  v = new_v, \
															  dU = new_dU, \
															  trace = new_trace);

					valShotAccuracy += shotAccuracy(torch.argmax(output, dim=2), label)/val_batches;
			print(valShotAccuracy, torch.mean(valShotAccuracy));
			val_errors.append(valShotAccuracy);
			# scheduler1.step(torch.mean(valShotAccuracy));
		
	
with torch.no_grad():
	error = 0;
	for idx, batch in tqdm(enumerate(test_iter)):
		input_total, labels = offset(batch);
		new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
		new_v, new_h, new_dU, new_trace, (_, output) = model.eval().forward(\
												  x = input_total,\
												  h = new_h, \
												  v = new_v, \
												  dU = new_dU, \
												  trace = new_trace);

		error += 1.0*torch.mean(torch.argmax(output, dim=2)!=labels)/len(test_iter);
print(error);