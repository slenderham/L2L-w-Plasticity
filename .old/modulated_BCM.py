import torch
import math
from utils.activations import SaturatingPoisson, Swish, SoftSignReLU
from utils.dropouts import embedded_dropout, LockedDropout, WeightDrop

'''
 hyperparameters:

 tau of post / pre trace
 a of post / pre trace
 tau of STDP (weight decay)

 the integrated total change in trace by one spike should integrate to 1 (I think)

 so a = 1/[sum_i tau^i] = (1-tau)/tau

 that's two less hyperparams to optimize

 add Dropout

'''




class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, mod_rank, bias=True):

		"""
		GRU with a ReLU

		v_t = (1-z) * v_{t-1} + z * (Wx_{t-1} + (U + U_plastic)(r * h_{t-1}) + b)
		h_t = [v_t]_+
		z, r = sigmoid(Wx + Uh + b)

		"""

		super(SGRUCell, self).__init__();

		self.input_dim = input_dim;
		self.hidden_dim = hidden_dim;
		self.bias = bias;

		# input-hidden weights
		self.x2h = torch.nn.Linear(input_dim, (4) * hidden_dim, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, (4) * hidden_dim, bias=bias)
		# layer norm
		self.lnx = torch.nn.LayerNorm(4*hidden_dim);
		self.lnh = torch.nn.LayerNorm(4*hidden_dim);

		# spiking function
		if (activation=="relu"):
			self.act = torch.nn.ReLU();
		elif (activation=="softplus"):
			self.act = torch.nn.Softplus();
		elif (activation=="tanh"):
			self.act = torch.nn.Tanh();
		elif (activation=="poisson"):
			self.act = SaturatingPoisson();
		else:
			self.act = Swish();
		# strength of weight modification
		self.alpha = torch.nn.Parameter(torch.rand(1, hidden_dim, 1)*0.001);
#		self.alpha = 0.01*torch.ones(1,1);

		# weight modification
		self.mod = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, mod_rank),
										 torch.nn.ReLU(),
										 torch.nn.Linear(mod_rank, self.hidden_dim));

		# time constant of STDP weight modification
		self.tau_U = torch.nn.Parameter(torch.randn(1, hidden_dim, 1)*math.sqrt(1/self.hidden_dim));

	def forward(self, x, h, v, dU, trace):

		trace_e, trace_E = trace;

		# preactivations
		Wx = (self.x2h(x));
		Wh = (self.h2h(h));

		# segment into gates: forget and reset gate for GRU, concurrent STDP modulation for the eligibility trace
		z = torch.sigmoid(Wx[:,:self.hidden_dim] + Wh[:,:self.hidden_dim]);
		r = torch.sigmoid(Wx[:,self.hidden_dim:2*self.hidden_dim] + Wh[:,self.hidden_dim:2*self.hidden_dim]);
		s = torch.sigmoid(Wx[:,2*self.hidden_dim:3*self.hidden_dim] + Wh[:,2*self.hidden_dim:3*self.hidden_dim]).unsqueeze(2);
		dv = Wx[:,3*self.hidden_dim:] + r * (Wh[:,3*self.hidden_dim:] + torch.bmm(torch.abs(self.alpha)*dU, h.unsqueeze(2)).squeeze(2));
		new_v = (1-z) * v + z * dv;

		new_h = self.act(new_v);

		m = self.mod(new_h);

		new_trace_e = (1-z)*trace_e + z*new_h**2;

		# clip weight modification between for stability

		new_trace_E = (1-s)*trace_E + s*\
			torch.bmm((new_h.unsqueeze(2))*(new_h.unsqueeze(2)-new_trace_e.unsqueeze(2)), h.unsqueeze(1));

		new_dU = dU+torch.sigmoid(self.tau_U)*m.unsqueeze(2)*new_trace_E;

		new_dU = torch.clamp(new_dU, -2, +2);

		return new_v, new_h, new_dU, (new_trace_e, new_trace_E);

	def get_init_states(self, batch_size=1):
		h_0 = torch.zeros(batch_size, self.hidden_dim);
		v_0 = torch.zeros(batch_size, self.hidden_dim);
		dU_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim);
		trace_e_0 = torch.zeros(batch_size, self.hidden_dim);
		trace_E_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim);
		return h_0, v_0, dU_0, (trace_e_0, trace_E_0);

class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers, mod_rank = 1, padding_idx=None, activation="swish"):
		super(SGRU, self).__init__();

		self.hidden_dim = hidden_dim;
		self.input_dim = input_dim;
		self.out_dim = out_dim;
		self.num_token = num_token;

		assert(in_type in ["continuous", "categorical", "binary"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

		if in_type=="categorical":
			self.encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
		else:
			self.encoder = None;

		self.rnns = torch.nn.ModuleList();
		self.rnns.append(SGRUCell(input_dim, hidden_dim, activation, mod_rank));
		for i in range(1, num_layers):
			self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation, mod_rank));

		if out_type=="continuous":
			self.decoder = torch.nn.Linear(hidden_dim, out_dim);
		elif out_type=="categorical":
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSoftmax(dim=1));
		else:
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSigmoid(dim=1));

		self.num_layers = num_layers;

		self.reset_parameter();

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
			new_v, new_h, new_dU, new_trace = self.rnns[i].forward(new_hs[i-1], h[i], v[i], dU[i], trace[i]);
			new_vs.append(new_v);
			new_hs.append(new_h);
			new_dUs.append(new_dU);
			new_traces.append(new_trace);

		output = self.decoder(new_hs[self.num_layers-1]);

		return new_vs, new_hs, new_dUs, new_traces, output;

	def get_init_states(self, batch_size=1):
		v_0 = [];
		h_0 = [];
		dU_0 = [];
		trace_0 = [];
		for rnn in self.rnns:
			h_i, v_i, dU_i, trace_i = rnn.get_init_states(batch_size);
			v_0.append(v_i);
			h_0.append(h_i);
			dU_0.append(dU_i);
			trace_0.append(trace_i);

		return h_0, v_0, dU_0, trace_0;

	def scale_grad(self):
		for rnn in self.rnns:
			rnn.tau_U.grad /= self.hidden_dim;
			rnn.alpha.grad /= self.hidden_dim;

	def detach(self, new_v, new_h, new_dU, new_trace):
		new_v[:] = [v.detach() for v in new_v];
		new_h[:] = [h.detach() for h in new_h];
		new_dU[:] = [dU.detach() for dU in new_dU];
		new_trace[:] = [[trace.detach() for trace in traces] for traces in new_trace];

	def reset_parameter(self):
		print(sum([p.numel() for p in self.parameters()]));
		for name, param in self.named_parameters():
			if "h2h.weight" in name:
				for i in range(3):
					torch.nn.init.orthogonal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
				#  torch.nn.init.normal_(param.data, mean=0, std=1/self.hidden_dim**0.5);
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "h2h.bias" in name :
				torch.nn.init.zeros_(param.data);
				param.data[:2 * self.hidden_dim] = 1;
			elif "mod" in name and "weight" in name:
				torch.nn.init.xavier_normal_(param.data);
			elif "mod" in name and "bias" in name:
				torch.nn.init.zeros_(param.data);
			elif "encoder.weight" in name:
				torch.nn.init.xavier_normal_(param.data);