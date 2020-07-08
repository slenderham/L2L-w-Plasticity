import torch

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
class STDP(torch.nn.Module):
	def __init__(self, hidden_dim):

		super(STDP, self).__init__();

		self.hidden_dim = hidden_dim;
		# parameters of STDP, tau : time constant of STDP window; a : magnitude of the weight modification
		self.tau = torch.clamp(0.02+0.001*torch.randn(1, 2*hidden_dim), 0, 1);
		self.a = (1-self.tau)/self.tau;

	def forward(self, trace, new_h):
		new_trace = torch.zeros_like(trace);
		new_trace[:,:self.hidden_dim] = (1-self.tau[:,:self.hidden_dim])*trace[:,:self.hidden_dim] + self.tau[:,:self.hidden_dim]*new_h;
		new_trace[:,self.hidden_dim:] = (1-self.tau[:,self.hidden_dim:])*trace[:,self.hidden_dim:] + self.tau[:,self.hidden_dim:]*new_h;

		return torch.mm((self.a[:,:self.hidden_dim]*new_h).t(), new_trace[:,:self.hidden_dim]) - torch.mm((self.a[:,self.hidden_dim:]*new_trace[:,self.hidden_dim:]).t(), new_h), new_trace;

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
		# layer normalization
		self.ln = torch.nn.LayerNorm(hidden_dim);

		# spiking function
		if (activation=="relu"):
			self.act = torch.nn.ReLU();
		elif (activation=="softplus"):
			self.act = torch.nn.Softplus();
		else:
			self.act = torch.nn.Tanh();

		# strength of weight modification
		self.alpha = torch.nn.Parameter(.0001 * torch.rand(hidden_dim, hidden_dim));

		# weight modification
		self.STDP = STDP(self.hidden_dim);

		# time constant of STDP weight modification
		self.tau_u = 0.01;

		self.reset_parameter();

	def reset_parameter(self):
		std = 1/(self.hidden_dim)**(1/2);
		for name, param in self.named_parameters():
			if "h2h.weight" in name:
				torch.nn.init.orthogonal_(param.data)*std;
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data)*std;
			elif "bias" in name :
				torch.nn.init.zeros_(param.data);
				param.data[self.hidden_dim:2 * self.hidden_dim] = 1;


	def forward(self, x, h, v, dU, trace):

		# preactivations
		Wx = self.x2h(x);
		Wh = self.h2h(h);

		# segment into gates
		z = torch.sigmoid(Wx[:,:self.hidden_dim] + Wh[:,:self.hidden_dim]);
		r = torch.sigmoid(Wx[:,self.hidden_dim:2*self.hidden_dim] + Wh[:,self.hidden_dim:2*self.hidden_dim]);
		dv = Wx[:,2*self.hidden_dim:] + r * (Wh[:,2*self.hidden_dim:] + torch.mm(h, (torch.abs(self.alpha)*dU).t()));
		new_v = self.ln((1-z) * v + z * dv);

		# switch to continuous
		new_h = self.act(new_v);

		# clip weight modification between [-1, +1] for stability
		ddU, new_trace = self.STDP(trace, new_h);

		new_dU = (1-self.tau_u)*dU+self.tau_u*ddU;

#		new_dU.data[new_dU>1] = 1;
#		new_dU.data[new_dU<-1] = -1;
		new_dU = torch.clamp(new_dU, -1, 1);

		return new_v, new_h, new_dU, new_trace;

	def get_init_states(self, scale=1):
		h_0 = torch.zeros(1, self.hidden_dim);
		v_0 = torch.zeros(1, self.hidden_dim) * scale;
		dU_0 = torch.zeros(self.hidden_dim, self.hidden_dim);
#		trace_0 = torch.clamp(0.5+torch.randn(1, 2*self.hidden_dim)*0.1, 0, 1);
		trace_0 = torch.zeros(1, 2*self.hidden_dim);
		return h_0, v_0, dU_0, trace_0;

class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers, activation="relu"):
		super(SGRU, self).__init__();

		assert(in_type in ["continuous", "categorical", "binary"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh"]), "please use a correct activation function";

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


	def detach(self, new_v, new_h, new_dU, new_trace):
		new_v[:] = [v.detach() for v in new_v];
		new_h[:] = [h.detach() for h in new_h];
		new_dU[:] = [dU.detach() for dU in new_dU];
		new_trace[:] = [trace.detach() for trace in new_trace];



