import torch
import math

# separate modules
# write weight clippers

class SurrHeaviside(torch.nn.Module):
	def __init__(self, threshold=0):
		super(SurrHeaviside, self).__init__();
		self.threshold = threshold;

	def forward(self, v):
		out = torch.zeros_like(v);
		out[v>self.threshold] = 1.0;
		return out;

class STDP(torch.nn.Module):
	def __init__(self, hidden_dim, tau_v):

		super(STDP, self).__init__();

		self.hidden_dim = hidden_dim;
		# parameters of STDP, tau : time constant of STDP window; a : magnitude of the weight modification
		self.tau_v = tau_v;

	def forward(self, trace, new_h):
		new_trace = (1-self.tau_v)*trace + self.tau_v*new_h;

		return torch.mm((new_h).t(), new_trace) - torch.mm((new_trace).t(), new_h), new_trace;


class RNNCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, bias=True):

		"""
		Vanilla RNN

		"""

		super(RNNCell, self).__init__();

		self.input_dim = input_dim;
		self.hidden_dim = hidden_dim;
		self.bias = bias;

		# input-hidden weights
		self.x2h = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, hidden_dim, bias=bias)

		# activation, default is ReLU
		if (activation=="relu"):
			self.act = torch.nn.ReLU();
		elif (activation=="softplus"):
			self.act = torch.nn.Softplus();
		else:
			self.act = torch.nn.Tanh();

		# strength of weight modification
		self.alpha = torch.nn.Parameter(torch.abs(torch.randn(1, hidden_dim)*math.sqrt(1/self.hidden_dim)));
#		self.alpha = torch.zeros(1,1);

		# time constant of STDP weight modification
		self.tau_u = 0.01;

		# time constant for membrane voltage
		self.tau_v = 0.1;

		# weight modification
		self.STDP = STDP(self.hidden_dim, self.tau_v);

		self.reset_parameter();

	def reset_parameter(self):
		for name, param in self.named_parameters():
			if "h2h.weight" in name:
				torch.nn.init.xavier_normal_(param.data)*1.5;
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "bias" in name :
				torch.nn.init.zeros_(param.data);

	def forward(self, x, h, v, dU, trace):
		# unpack activity
#		h = h.view(h.size(1), -1)
#		v = v.view(v.size(1), -1)

		# preactivations
		Wx = self.x2h(x);
		Wh = self.h2h(h);


		dv = Wx + Wh + torch.abs(self.alpha)*torch.mm(h, dU.t());
		new_v = (1-self.tau_v) * v + self.tau_v * dv;

		# spike in forward pass, surrogate gradient (sigmoid') in backward pass
		new_h = self.act(new_v);

		ddU, new_trace = self.STDP(trace, new_h);
		new_dU = (1-self.tau_u)*dU+self.tau_u*ddU;
		new_dU = torch.clamp(new_dU, -1, 1);

		return new_v, new_h, new_dU, new_trace;

	def get_init_states(self, scale=.1):
		h_0 = torch.zeros(1, self.hidden_dim);
		v_0 = torch.randn(1, self.hidden_dim) * scale;
		dU_0 = torch.zeros(self.hidden_dim, self.hidden_dim);
#		trace_0 = torch.clamp(0.5+torch.randn(1, 2*self.hidden_dim)*0.1, 0, 1);
		trace_0 = torch.zeros(1, self.hidden_dim);
		return h_0, v_0, dU_0, trace_0;

class RNN(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, activation):
		super(RNN, self).__init__();

		assert(in_type in ["continuous", "categorical", "binary"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";

		if in_type=="categorical":
			self.encoder = torch.Embedding(num_token, input_dim);
		else:
			self.encoder = None;

		self.rnn = RNNCell(input_dim, hidden_dim, activation);


		if out_type=="continuous":
			self.decoder = torch.nn.Linear(hidden_dim, out_dim);
		elif out_type=="categorical":
			self.decoder = torch.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSoftmax());
		else:
			self.decoder = torch.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSigmoid());

	def forward(self, x, h, v, dU, trace):
		if self.encoder!=None:
			new_v, new_h, new_dU, new_trace = self.rnn.forward(self.encoder(x), h[0], v[0], dU[0], trace[0]);
		else:
			new_v, new_h, new_dU, new_trace = self.rnn.forward(x, h[0], v[0], dU[0], trace[0]);

		output = self.decoder(new_h);

		return [new_v], [new_h], [new_dU], [new_trace], output;

	def get_init_states(self):
		v_0 = [];
		h_0 = [];
		dU_0 = [];
		trace_0 = [];

		h_i, v_i, dU_i, trace_i = self.rnn.get_init_states();
		v_0.append(v_i);
		h_0.append(h_i);
		dU_0.append(dU_i);
		trace_0.append(trace_i);

		return h_0, v_0, dU_0, trace_0;


