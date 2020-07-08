import torch
import math
from activations import SaturatingPoisson, Swish, Spike, TernaryTanh, kWTA, BipolarWrapper
from dropouts import embedded_dropout, LockedDropout, WeightDrop


class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, mod_rank, positive_proportion, bias=True):

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

		# input-hidden weights
		self.x2h = torch.nn.Linear(input_dim, (4) * hidden_dim, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, (4) * hidden_dim, bias=bias)

		# layer norm
		self.lnx = torch.nn.LayerNorm(4*hidden_dim);
		self.lnh = torch.nn.LayerNorm(4*hidden_dim);

		# spiking function
		if (activation=="relu"):
			self.act = BipolarWrapper(torch.nn.ReLU(), positive_proportion=positive_proportion);
		elif (activation=="softplus"):
			self.act = BipolarWrapper(torch.nn.Softplus(), positive_proportion=positive_proportion);
		elif (activation=="tanh"):
			self.act = torch.nn.Tanh();
		elif (activation=="poisson"):
			self.act = SaturatingPoisson();
		else:
			self.act = BipolarWrapper(Swish(), positive_proportion=positive_proportion);
		# strength of weight modification
		self.alpha = torch.nn.Parameter(0.1*torch.rand(1, self.hidden_dim, 1)*math.sqrt(1/self.hidden_dim));

		# weight modification
		self.h2mod = torch.nn.Linear(self.hidden_dim, mod_rank);
		self.kWTA = kWTA(0.25);
		self.mod2h = torch.nn.Linear(mod_rank, self.hidden_dim);

		# time constant of STDP weight modification
		self.tau_U = torch.nn.Parameter(torch.log(20.0+torch.rand(1, self.hidden_dim, 1)*40.0));

		self.EI_mask = torch.cat([
									torch.ones(1, round(self.hidden_dim*self.positive_proportion), 1), 
									-torch.ones(1, self.hidden_dim-round(self.hidden_dim*self.positive_proportion), 1),
								 ], dim=1);
		
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

		f = torch.sigmoid(Wx[:,:self.hidden_dim] + Wh[:,:self.hidden_dim]);
		i = torch.sigmoid(Wx[:,self.hidden_dim:2*self.hidden_dim] + Wh[:,self.hidden_dim:2*self.hidden_dim]);
		s = torch.sigmoid(Wx[:,2*self.hidden_dim:3*self.hidden_dim] + Wh[:,2*self.hidden_dim:3*self.hidden_dim]).unsqueeze(2);
		dv = Wx[:,3*self.hidden_dim:] + Wh[:,3*self.hidden_dim:] + torch.bmm((self.EI_mask*torch.abs(self.alpha))*dU, h.unsqueeze(2)).squeeze(2);
		v = f * v + i * dv;

		new_h = self.act(v);

		h2m = self.h2mod(new_h);
		mWTA = self.kWTA(mod);
		m2h = self.mod2h(mWTA);

		new_trace_e = f*trace_e + i*new_h;

		new_trace_E = (1-s)*trace_E + s*(\
 			torch.bmm(new_h.unsqueeze(2), trace_e.unsqueeze(1)) - torch.bmm(new_trace_e.unsqueeze(2), h.unsqueeze(1)));

		dU = torch.sigmoid(self.tau_U)*dU+m2h.unsqueeze(2)*trace_E;
		dU.data.clamp_(-5, +5);

		return v, new_h, dU, (new_trace_e, new_trace_E);

	def reset_parameter(self):
		for name, param in self.named_parameters():
			setattr(self, name.replace(".", "_"), param);
			print(name);
			if "h2h.weight" in name:
				for i in range(4):
 					torch.nn.init.orthogonal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
 					# torch.nn.init.normal_(param.data, mean=0, std=1.5/self.hidden_dim**0.5);
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "x2h.bias" in name:
				torch.nn.init.zeros_(param.data);
			elif "h2h.bias" in name :
				torch.nn.init.zeros_(param.data);
				param.data[0:self.hidden_dim] = torch.log(1.0+torch.rand(self.hidden_dim)*20.0);
				param.data[1*self.hidden_dim:2*self.hidden_dim] = -torch.log(1.0+torch.rand(self.hidden_dim)*20.0);
				param.data[2*self.hidden_dim:3*self.hidden_dim] = -torch.log(10.0+torch.rand(self.hidden_dim)*30.0);
			elif "mod" in name and "weight" in name:
				torch.nn.init.xavier_normal_(param.data);
			elif "mod" in name and "bias" in name:
				torch.nn.init.zeros_(param.data);

	def get_init_states(self, batch_size, device):
		h_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		v_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		dU_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim).to(device);
		trace_e_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		trace_E_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim).to(device);
		return h_0, v_0, dU_0, (trace_e_0, trace_E_0);

class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, positive_proportion = 0.8, num_layers=1, \
				  dropout_e = 0, dropout_i = 0, dropout_h = 0, dropout_o = 0, dropout_w = 0,\
				  mod_rank = 1, padding_idx=None, activation="swish", tie_weight=True):

		super(SGRU, self).__init__();

		self.hidden_dim = hidden_dim;
		self.input_dim = input_dim;
		self.out_dim = out_dim;
		self.num_token = num_token;
		self.in_type = in_type;
		self.out_type = out_type;
		self.tie_weight = tie_weight if (in_type=="categorical" and out_type=="categorical") else False;

		self.dropout_e = dropout_e;
		self.dropout_i = dropout_e;
		self.dropout_h = dropout_h;
		self.dropout_o = dropout_o;
		self.dropout_w = dropout_w;

		assert(in_type in ["continuous", "categorical", "binary"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

		if in_type=="categorical":
			self.encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
		else:
			self.encoder = None;

		self.rnns = [];
		self.rnns.append(SGRUCell(input_dim, hidden_dim, activation, mod_rank, positive_proportion));

		if (self.tie_weight):
			for i in range(1, num_layers-1):
				self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation, mod_rank, positive_proportion));
			self.rnns.append(SGRUCell(hidden_dim, input_dim, activation, mod_rank, positive_proportion));
		else:
			for i in range(1, num_layers):
				self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation, mod_rank));

		if (self.dropout_w):
 			self.rnns = [WeightDrop(self.rnns[l], ["h2h_weight", "h2mod_0_weight", "h2mod_2_weight"], self.dropout_w) for l in range(num_layers)];

		self.rnns = torch.nn.ModuleList(self.rnns);

		if out_type=="continuous":
			self.decoder = torch.nn.Linear(hidden_dim, out_dim);
		elif out_type=="categorical":
			if (self.tie_weight):
				self.decoder = torch.nn.Sequential(torch.nn.Linear(input_dim, out_dim), torch.nn.LogSoftmax(dim=2));
				self.decoder.weight = self.encoder.weight;
			else:
				self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSoftmax(dim=2));
		else:
			self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSigmoid(dim=2));

		self.num_layers = num_layers;

		# dropouts
		self.locked_drop = LockedDropout();

		self.reset_parameter();

	def forward(self, x, h, v, dU, trace):

		# size of x is seq_len X batch_size X input_dimension

		if self.in_type=="categorical":
			x = embedded_dropout(self.encoder, x, dropout=self.dropout_e if self.training else 0);

		prev_out = self.locked_drop(x, self.dropout_i);

		for l, rnn in enumerate(self.rnns):
			v[l], h[l], dU[l], trace[l], prev_out = rnn.forward(prev_out, h[l], v[l], dU[l], trace[l]);

			if l!=self.num_layers-1:
				prev_out = self.locked_drop(prev_out, self.dropout_h);
			else:
				prev_out = self.locked_drop(prev_out, self.dropout_o);

		return v, h, dU, trace, (prev_out, self.decoder(prev_out));

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
			# rnn.module.tau_U.grad /= self.hidden_dim;
			rnn.module.alpha.grad /= self.hidden_dim;

	def detach(self, new_v, new_h, new_dU, new_trace):
		new_v[:] = [v.detach() for v in new_v];
		new_h[:] = [h.detach() for h in new_h];
		new_dU[:] = [dU.detach() for dU in new_dU];
		new_trace[:] = [[trace.detach() for trace in traces] for traces in new_trace];

	def reset_parameter(self):
		print(sum([p.numel() for p in self.parameters()])- (self.decoder.weight.numel() if self.tie_weight else 0));

		if self.in_type=="categorical":
			torch.nn.init.xavier_uniform_(self.encoder.weight.data);

		if self.out_type!="categorical":
			torch.nn.init.xaview_normal_(self.decoder.weight.data);
			torch.nn.init.zeros_(self.decoder.bias.data);