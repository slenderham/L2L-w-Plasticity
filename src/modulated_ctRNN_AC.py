import torch
import math
from activations import SaturatingPoisson, Swish, Spike, TernaryTanh, kWTA, NormalizeWeight
from dropouts import embedded_dropout, LockedDropout, WeightDrop


class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, mod_rank, clip_val=1.0, bias=True):

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
		self.mod_rank = mod_rank;
		self.clip_val = clip_val;

		# input-hidden weights
		self.x2h = torch.nn.Linear(input_dim, hidden_dim + mod_rank, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, hidden_dim + mod_rank, bias=bias)

		# layer norm
		self.lnx = torch.nn.LayerNorm(hidden_dim + mod_rank);
		self.lnh = torch.nn.LayerNorm(hidden_dim + mod_rank);

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
		self.alpha = torch.nn.Parameter(0.01*torch.ones(1, self.hidden_dim, self.hidden_dim));
		self.mod2h = torch.nn.Parameter(torch.ones(1, self.hidden_dim, self.mod_rank)/self.mod_rank);

		# time constant of STDP weight modification
		self.tau_v = torch.nn.Parameter(-torch.log(1+5*torch.rand(1, self.hidden_dim)));
		self.tau_U = torch.nn.Parameter(-2.0*torch.ones(1, self.hidden_dim, self.hidden_dim));
		self.tau_E = torch.nn.Parameter(-1.0*torch.ones(1, self.hidden_dim, self.hidden_dim));		
		self.reset_parameter();

	def forward(self, x, h, v, dU, trace):
		curr_out = [];
		for c in range(x.shape[0]):
			v, h, dU, trace = self._forward_step(x[c], h, v, dU, trace);
			curr_out.append(h);
		return v, h, dU, trace, torch.stack(curr_out);

	def _forward_step(self, x, h, v, dU, trace):
		trace_e, trace_E = trace;

		# preactivations
		Wx = self.lnx(self.x2h(x));
		Wh = self.lnh(self.h2h(h));

		# segment into gates: forget and reset gate for GRU, concurrent STDP modulation for the eligibility trace
		# clip weight modification between for stability (only when it's used)

		dv = Wx[:,:self.hidden_dim] + Wh[:,:self.hidden_dim] + torch.bmm(torch.relu(self.alpha)*dU, h.unsqueeze(2)).squeeze(2);
		v = (1-torch.sigmoid(self.tau_v)) * v + torch.sigmoid(self.tau_v) * dv;

		new_h = self.act(v);
		mod = (self.mod2h @ (self.act(Wx[:,self.hidden_dim:]+Wh[:,self.hidden_dim:]).unsqueeze(2))).squeeze(2)#.sum(dim=-1).unsqueeze(-1).unsqueeze(-1);

		new_trace_e = (1-torch.sigmoid(self.tau_v))*trace_e + torch.sigmoid(self.tau_v)*h;
		new_trace_E = (1-torch.sigmoid(self.tau_E))*trace_E + torch.sigmoid(self.tau_E)*(torch.bmm(new_h.unsqueeze(2), trace_e.unsqueeze(1)) - torch.bmm(new_trace_e.unsqueeze(2), h.unsqueeze(1)));
		dU = (1-torch.sigmoid(self.tau_U))*dU+torch.sigmoid(self.tau_U)*mod.unsqueeze(-1)*new_trace_E;
		dU = torch.max( 
					torch.min(dU, torch.relu(self.clip_val-self.h2h.weight[:self.hidden_dim,:])/(torch.relu(self.alpha)+1e-8)), 
						-torch.relu(self.clip_val+self.h2h.weight[:self.hidden_dim,:])/(torch.relu(self.alpha)+1e-8)
					  );

		return v, new_h, dU, (new_trace_e, new_trace_E);

	def reset_parameter(self):
		for name, param in self.named_parameters():
			setattr(self, name.replace(".", "_"), param);
			print(name);
			if "h2h.weight" in name:
				torch.nn.init.orthogonal_(param.data[:self.hidden_dim,:]);
				torch.nn.init.xavier_normal_(param.data[self.hidden_dim:,:]);
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "x2h.bias" in name:
				torch.nn.init.zeros_(param.data);
			elif "h2h.bias" in name :
				torch.nn.init.zeros_(param.data);

	def get_init_states(self, batch_size, device):
		h_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		v_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		dU_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim).to(device);
		trace_e_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		trace_E_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim).to(device);
		return h_0, v_0, dU_0, (trace_e_0, trace_E_0);


class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers=1, \
				  mod_rank = 1, padding_idx=None, activation="swish", tie_weight=True):

		super(SGRU, self).__init__();

		self.hidden_dim = hidden_dim;
		self.input_dim = input_dim;
		self.out_dim = out_dim;
		self.num_token = num_token;
		self.in_type = in_type;
		self.out_type = out_type;

		assert(in_type in ["continuous", "categorical", "image"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

		if in_type=="categorical":
			self.encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
		elif in_type=="image":
			self.conv1 = nn.Conv2d(1, 16, 8, 4);
			self.conv2 = nn.Conv2d(16, 32, 4, 2);
			self.fc1 = nn.Linear(128, self.input_dim-self.out_dim-1);
			def encode(x):
				x = torch.relu(self.conv1(x))
				x = torch.relu(self.conv2(x))
				x = torch.flatten(x, 1);
				x = torch.relu(self.fc1(x));
				return x;
			self.encoder = encode;

		self.rnns = [];
		self.rnns.append(SGRUCell(input_dim, hidden_dim, activation, mod_rank));

		for i in range(1, num_layers):
			self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation, mod_rank));

		self.rnns = torch.nn.ModuleList(self.rnns);

		if out_type=="continuous":
			self.decoder = torch.nn.Linear(hidden_dim, out_dim);
		elif out_type=="categorical":
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSoftmax(dim=2));
		else:
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSigmoid(dim=2));

		self.num_layers = num_layers;
		self.critic = torch.nn.Linear(hidden_dim*num_layers, 1);

		# dropouts
		self.reset_parameter();

	def forward(self, x, h, v, dU, trace):

		# size of x is seq_len X batch_size X input_dimension

		if self.in_type=="categorical":
			prev_out = self.encoder(x);
		elif self.in_type=="image":
			prev_out = torch.stack([self.encoder(stepX) for stepX in x]);
		else:
			prev_out = x;

		for l, rnn in enumerate(self.rnns):
			v[l], h[l], dU[l], trace[l], prev_out = rnn.forward(prev_out, h[l], v[l], dU[l], trace[l]);

		action_logit = self.decoder(prev_out);
		value = self.critic(prev_out);

		return v, h, dU, trace, (prev_out, action_logit, value);

	def get_init_states(self, batch_size, device):
		v_0 = [];
		h_0 = [];
		dU_0 = [];
		trace_0 = [];
		for rnn in self.rnns:
			h_i, v_i, dU_i, trace_i = rnn.get_init_states(batch_size, device);
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

		if self.in_type=="categorical":
			torch.nn.init.xavier_uniform_(self.encoder.weight.data);
			
		if self.out_type=="categorical" or self.out_type=="binary":
			torch.nn.init.xavier_normal_(self.decoder[0].weight.data);
			torch.nn.init.zeros_(self.decoder[0].bias.data);
		elif self.out_type=="continuous":
			torch.nn.init.xavier_normal_(self.decoder.weight.data);
			torch.nn.init.zeros_(self.decoder.bias.data);
