import torch
import math

'''
 hyperparameters:

 tau of post / pre trace
 a of post / pre trace
 upper lower bound of dU
 alpha of weight modification
 tau of STDP (weight decay)

 the integrated total change in trace by one spike should integrate to 1 (I think)

 so a = 1/[sum_i tau^i] = (1-tau)/tau

 that's two less hyperparams to optimize

 add Dropout

'''

class SaturatingPoisson(torch.nn.Module):
	def __init__(self):
		super(SaturatingPoisson, self).__init__();

	def forward(self, x):
		return 1-torch.exp(-torch.relu(x));


class STDP(torch.nn.Module):
	def __init__(self, hidden_dim):

		super(STDP, self).__init__();

		self.hidden_dim = hidden_dim;
		# parameters of STDP, tau : time constant of STDP window; a : magnitude of the weight modification

	def forward(self, trace, new_h, r, z):

		ddU = torch.mm((new_h).t(), trace) - torch.mm(trace.t(), new_h);
		new_trace = (1-z)*trace + z*r*new_h;

		return ddU, new_trace;

class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, bias=True):

		"""
		GRU with a spiking activation function

		v_t = (1-z) * v_{t-1} + z * (Wx_{t-1} + (U + U_plastic)(r * h_{t-1}) + b)
		h_t = v_t > threshold
		z, r = sigmoid(Wx + Uh + b)

		"""

		super(SGRUCell, self).__init__();

		self.input_dim = input_dim;
		self.hidden_dim = hidden_dim;
		self.bias = bias;

		# input-hidden weights
		self.x2h = torch.nn.Linear(input_dim, (3) * hidden_dim, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, (3) * hidden_dim, bias=bias)
		# layer norm
		self.ln = torch.nn.LayerNorm(hidden_dim);

		# activation, default is ReLU
		if (activation=="relu"):
			self.act = torch.nn.ReLU();
		elif (activation=="softplus"):
			self.act = torch.nn.Softplus();
		elif (activation=="tanh"):
			self.act = torch.nn.Tanh();
		else:
			self.act = SaturatingPoisson();

		# strength of weight modification
		self.alpha = torch.nn.Parameter(torch.rand(1, hidden_dim)*math.sqrt(1/self.hidden_dim));
#		self.alpha = torch.zeros(1,1);

		# weight modification
		self.STDP = STDP(self.hidden_dim);

		# time constant of STDP weight modification
		self.tau_u = 0.01;

		self.reset_parameter();

	def reset_parameter(self):
		for name, param in self.named_parameters():
			if "h2h.weight" in name:
				for i in range(3):
					torch.nn.init.orthogonal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
#					torch.nn.init.normal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:], mean=0, std=2/self.hidden_dim**0.5)
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "bias" in name :
				torch.nn.init.zeros_(param.data);
				param.data[:2 * self.hidden_dim] = 3;

	def forward(self, x, h, v, dU, trace):

		# preactivations
		Wx = self.x2h(x);
		Wh = self.h2h(h);

		# segment into gates
		z = torch.sigmoid(Wx[:,:self.hidden_dim] + Wh[:,:self.hidden_dim]);
		r = torch.sigmoid(Wx[:,self.hidden_dim:2*self.hidden_dim] + Wh[:,self.hidden_dim:2*self.hidden_dim]);
		dv = Wx[:,2*self.hidden_dim:] + r * (Wh[:,2*self.hidden_dim:] + torch.abs(self.alpha)*torch.mm(h, dU.t()));
		new_v = (1-z) * v + z * dv;

		# switch to continuous
		new_h = self.act(new_v);

		# clip weight modification between [-1, +1] for stability
		ddU, new_trace = self.STDP(trace = trace, new_h = new_h, r = r, z = z);

		new_dU = (1-self.tau_u)*dU+self.tau_u*ddU;

		new_dU = torch.clamp(new_dU, -1, 1);

		return new_v, new_h, new_dU, new_trace;

	def get_init_states(self, scale=1):
		h_0 = torch.zeros(1, self.hidden_dim);
		v_0 = torch.zeros(1, self.hidden_dim) * scale;
		dU_0 = torch.zeros(self.hidden_dim, self.hidden_dim);
		trace_0 = torch.zeros(1, self.hidden_dim);
#		trace_0 = torch.rand(1, self.hidden_dim);
		return h_0, v_0, dU_0, trace_0;

class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers, activation="relu"):
		super(SGRU, self).__init__();

		assert(in_type in ["continuous", "categorical", "binary"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh", "poisson"]), "please use a correct activation function";

		if in_type=="categorical":
			self.encoder = torch.nn.Embedding(num_token, input_dim);
		else:
			self.encoder = None;

		self.rnns = torch.nn.ModuleList();
		self.rnns.append(SGRUCell(input_dim, hidden_dim, activation));

		for i in range(1, num_layers):
			self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation));

		if out_type=="continuous":
			self.decoder = torch.nn.Linear(hidden_dim, out_dim);
		elif out_type=="categorical":
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSoftmax(dim=1));
		else:
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSigmoid(dim=1));

		self.num_layers = num_layers

	def forward(self, x, h, v, dU, trace):
		new_vs = [];
		new_hs = [];
		new_dUs = [];
		new_traces = [];

		if self.encoder!=None:
			new_v, new_h, new_dU, new_trace = self.rnns[0].forward(self.encoder(x), h[0], v[0], dU[0], trace[0]);
		else:
			new_v, new_h, new_dU, new_trace = self.rnns[0].forward(x, h[0], v[0], dU[0], trace[0]);

		new_vs.append(new_v);
		new_hs.append(new_h);
		new_dUs.append(new_dU);
		new_traces.append(new_trace);

		for i in range(1, self.num_layers):
			new_v, new_h, new_dU, new_trace = self.rnns[i].forward(new_h, h[i], v[i], dU[i], trace[i]);
			new_vs.append(new_v);
			new_hs.append(new_h);
			new_dUs.append(new_dU);
			new_traces.append(new_trace);

		output = self.decoder(new_hs[self.num_layers-1]);

		return new_vs, new_hs, new_dUs, new_traces, output;

	def get_init_states(self):
		v_0 = [];
		h_0 = [];
		dU_0 = [];
		trace_0 = [];
		for rnn in self.rnns:
			h_i, v_i, dU_i, trace_i = rnn.get_init_states();
			v_0.append(v_i);
			h_0.append(h_i);
			dU_0.append(dU_i);
			trace_0.append(trace_i);

		return h_0, v_0, dU_0, trace_0;



