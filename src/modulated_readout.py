import torch
import math
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

class SaturatingPoisson(torch.nn.Module):
	def __init__(self):
		super(SaturatingPoisson, self).__init__();

	def forward(self, x):
		return 1-torch.exp(-torch.relu(x));

class Swish(torch.nn.Module):
	def __init__(self):
		super(Swish, self).__init__();

	def forward(self, x):
		return x*torch.sigmoid(x);


class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, bias=True):

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
		self.x2h = torch.nn.Linear(input_dim, (3) * hidden_dim, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, (3) * hidden_dim, bias=bias)
		# layer norm
		self.ln = torch.nn.LayerNorm(hidden_dim);

		# spiking function
		if (activation=="relu"):
			self.act = torch.nn.LeakyReLU();
		elif (activation=="softplus"):
			self.act = torch.nn.Softplus();
		elif (activation=="tanh"):
			self.act = torch.nn.Tanh();
		elif (activation=="poisson"):
			self.act = SaturatingPoisson();
		else:
			self.act = Swish();

		self.reset_parameter();

	def reset_parameter(self):
		print(self.named_parameters);
		for name, param in self.named_parameters():
			if "h2h.weight" in name:
				torch.nn.init.orthogonal_(param.data)
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data)
			elif "bias" in name :
				torch.nn.init.zeros_(param.data)
				param.data[:2 * self.hidden_dim] = 1;

	def forward(self, x, h, v):

		# preactivations
		Wx = self.x2h(x);
		Wh = self.h2h(h);

		# segment into gates
		z = torch.sigmoid(Wx[:,:self.hidden_dim] + Wh[:,:self.hidden_dim]);
		r = torch.sigmoid(Wx[:,self.hidden_dim:2*self.hidden_dim] + Wh[:,self.hidden_dim:2*self.hidden_dim]);
		dv = Wx[:,2*self.hidden_dim:] + r * Wh[:,2*self.hidden_dim:];
		new_v = (1-z) * v + z * dv;

		# switch to continuous
		new_h = self.act(new_v);

		return new_v, new_h;

	def get_init_states(self, scale=1):
		h_0 = torch.zeros(1, self.hidden_dim);
		v_0 = torch.zeros(1, self.hidden_dim) * scale;
		return h_0, v_0;

class PReadOut(torch.nn.Module):
	def __init__(self, input_dim, out_dim, bias=True, out_type="categorical"):
		super(PReadOut, self).__init__();
		self.input_dim = input_dim;
		self.out_dim = out_dim;

		self.h2o = torch.nn.Linear(input_dim, out_dim);

		self.tau_pre = torch.nn.Parameter(torch.randn(1, input_dim)-3);
		self.tau_post = torch.nn.Parameter(torch.randn(1, out_dim)-3);
		self.tau_U = torch.nn.Parameter(torch.randn(out_dim, 1)-3);
		self.tau_STDP = torch.nn.Parameter(torch.randn(out_dim, 1)-3);
		self.alpha = torch.nn.Parameter(torch.randn(1, out_dim)-3);

		if out_type=="continuous":
			self.act = torch.nn.Identity();
		elif out_type=="categorical":
			self.act = torch.nn.Softmax(dim=1);
		else:
			self.act = torch.nn.Sigmoid(dim=1);

	def forward(self, x, dU, trace, mod):

		trace_pre, trace_post, trace_STDP = trace;

		new_dU = (1-torch.sigmoid(self.tau_U))*dU + torch.sigmoid(self.tau_U)*torch.tanh(mod.t())*trace_STDP;

		o = self.act(self.h2o(x) + torch.sigmoid(self.alpha)*torch.mm(x, new_dU.t()));

		new_trace_pre = (1-torch.sigmoid(self.tau_pre))*trace_pre + torch.sigmoid(self.tau_pre)*x;
		new_trace_post =(1-torch.sigmoid(self.tau_post))*trace_post + torch.sigmoid(self.tau_post)*o;
		new_trace_STDP = (1-torch.sigmoid(self.tau_STDP))*trace_STDP + torch.sigmoid(self.tau_STDP)*(torch.mm(o.t(), trace_pre) - torch.mm(trace_post.t(), x));

		return o, new_dU, (new_trace_pre, new_trace_post, new_trace_STDP);


class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers, activation="relu"):
		super(SGRU, self).__init__();

		assert(in_type in ["continuous", "categorical", "binary"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

		self.num_layers = num_layers;
		self.out_dim = out_dim;
		self.input_dim = input_dim;
		self.hidden_dim = hidden_dim;

		if in_type=="categorical":
			self.encoder = torch.nn.Embedding(num_token, input_dim);
		else:
			self.encoder = None;

		self.rnns = torch.nn.ModuleList();
		self.rnns.append(SGRUCell(input_dim, hidden_dim, activation));
		for i in range(1, num_layers):
			self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation));

		self.decoder = PReadOut(input_dim=hidden_dim, out_dim=out_dim, out_type=out_type)
		self.mod = torch.nn.Linear(hidden_dim, out_dim);

	def forward(self, x, h, v, dU, trace):

		new_vs = [];
		new_hs = [];

		if self.encoder!=None:
			new_v, new_h = self.rnns[0].forward(self.encoder(x), h[0], v[0]);
		else:
			new_v, new_h = self.rnns[0].forward(x, h[0], v[0]);

		new_vs.append(new_v);
		new_hs.append(new_h);

		for i in range(1, self.num_layers):
			new_v, new_h = self.rnns[i].forward(new_hs[i-1], h[i], v[i]);
			new_vs.append(new_v);
			new_hs.append(new_h);

		mod = self.mod(new_hs[self.num_layers-1]);

		output, new_dU, new_trace = self.decoder(new_hs[self.num_layers-1], dU, trace, mod);

		return new_vs, new_hs, new_dU, new_trace, output;

	def get_init_states(self):
		v_0 = [];
		h_0 = [];

		for rnn in self.rnns:
			h_i, v_i = rnn.get_init_states();
			v_0.append(v_i);
			h_0.append(h_i);

		dU_0 = torch.zeros(self.out_dim, self.hidden_dim);
		trace_pre_0 = torch.zeros(1, self.hidden_dim);
		trace_post_0 = torch.zeros(1, self.out_dim);
		trace_STDP_0 = torch.zeros(self.out_dim, self.hidden_dim);

		return h_0, v_0, dU_0, (trace_pre_0, trace_post_0, trace_STDP_0);



