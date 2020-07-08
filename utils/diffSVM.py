import torch

def max_margin(logit, target, weight):
	t = torch.ones_like(logit)*(-1);
	t[torch.arange(target.shape[0]), target] = +1;
	obj = torch.sum(torch.relu(1-logit*target, 0)**2);
	return obj;


class SVM(torch.nn.Module):
	def __init__(self, input_size, output_size):
		self.input_size = input_size;
		self.output_size = output_size;

		self.linear = torch.nn.Linear(input_size, output_size);

	def forward(self, x):
		return self.linear(x);

class BiSVM(torch.nn.Module):
	def __init__(self, input_size, output_size):
		self.input_size = input_size;
		self.output_size = output_size;

		self.leftLinear = torch.nn.Parameters(torch.randn(output_size, input_size)/torch.sqrt(input_size));
		self.rightLinear = torch.nn.Parameters(torch.randn(input_size, output_size)/torch.sqrt(output_size));

	def forward(self, x):
		return torch.matmul(self.leftLinear, torch.matmul(x, self.rightLinear)).diagonal(dim1=1,dim2=2);

optimizer = torch.optim.SGD(weight_decay=0.5);
