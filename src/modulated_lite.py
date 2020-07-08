import torch
import math
from activations import SaturatingPoisson, Swish, Spike, TernaryTanh, kWTA, NormalizeWeight
from dropouts import embedded_dropout, LockedDropout, WeightDrop


class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, clip_val=1.0, bias=True):

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
		self.clip_val = clip_val;

		# input-hidden weights
		self.x2h = torch.nn.utils.weight_norm(torch.nn.Linear(input_dim, (3) * (hidden_dim), bias=bias))
		# hidden-hidden weights
		self.h2h = torch.nn.utils.weight_norm(torch.nn.Linear(hidden_dim, (3) * (hidden_dim), bias=bias))

		self.alpha = torch.nn.Parameter(0.1*torch.ones(1, self.hidden_dim, self.hidden_dim));
		self.h2mod = torch.nn.Linear(hidden_dim, 4, bias=bias);

		# self.lnx = torch.nn.LayerNorm((2) * (hidden_dim) + 4);
		# self.lnh = torch.nn.LayerNorm((2) * (hidden_dim) + 4);

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

		# time constant of STDP weight modification
		self.mod_U_fan_out_weight = torch.nn.Parameter(torch.ones(1, self.hidden_dim, self.hidden_dim));
		self.mod_U_fan_out_bias = torch.nn.Parameter(torch.zeros(1, self.hidden_dim, self.hidden_dim));
		self.reset_parameter();

	def forward(self, x, h, v, dU, trace):
		curr_out = [];
		for c in range(x.shape[0]):
			v, h, dU, trace = self._forward_step(x[c], h, v, dU, trace);
			curr_out.append(h);
		return v, h, dU, trace, torch.stack(curr_out);

	def _forward_step(self, x, h, v, dU, trace):
		trace_e, trace_E = trace;

		# split into gates
		z_x, r_x, dv_x= torch.split(self.x2h(x), [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1);
		z_h, r_h, dv_h= torch.split(self.h2h(h), [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1);

		z = torch.sigmoid(z_x+z_h);
		r = torch.sigmoid(r_x+r_h);
		dv = dv_x + (dv_h + torch.bmm(self.alpha*dU, h.unsqueeze(2)).squeeze(2));
		v = (1-z) * v + z * dv;

		new_h = self.act(v);

		tau_e, tau_E, tau_U, mod_U = torch.split(self.h2mod(new_h), [1,1,1,1], dim=-1);

		mod_U = self.act(mod_U).unsqueeze(-1);
		tau_e = torch.sigmoid(tau_e);
		tau_E = torch.sigmoid(tau_E.unsqueeze(-1));
		tau_U = torch.sigmoid(tau_U.unsqueeze(-1));

		new_trace_e = (1-tau_e)*trace_e + tau_e*h;
		new_trace_E = (1-tau_E)*trace_E + tau_E*(torch.bmm(new_h.unsqueeze(2), new_trace_e.unsqueeze(1)) - torch.bmm(new_trace_e.unsqueeze(2), new_h.unsqueeze(1)));
		dU = (1-tau_U)*dU+tau_U*torch.nn.functional.softshrink(mod_U*self.mod_U_fan_out_weight+self.mod_U_fan_out_bias)*new_trace_E;
		dU = torch.max( 
					torch.min(dU, torch.relu(self.clip_val-self.h2h.weight[self.hidden_dim:2*self.hidden_dim,:])/(self.alpha+1e-5)), 
						-torch.relu(self.clip_val+self.h2h.weight[self.hidden_dim:2*self.hidden_dim,:])/(self.alpha+1e-5),
					  );
		
		return v, new_h, dU, (new_trace_e, new_trace_E);

	def reset_parameter(self):
		for name, param in self.named_parameters():
			setattr(self, name.replace(".", "_"), param);
			print(name);
			if "h2h.weight" in name:
				for i in range(3):
					torch.nn.init.orthogonal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
			elif "x2h.weight" in name:
				for i in range(3):
					torch.nn.init.xavier_uniform_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
			elif "x2h.bias" in name:
				torch.nn.init.zeros_(param.data);
			elif "h2h.bias" in name :
				torch.nn.init.zeros_(param.data);
				param.data[:self.hidden_dim] = -1;
			elif "h2mod" in name and "weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "h2mod" in name and "bias" in name:
				param.data[0] = 0;
				param.data[1:3] = -2;
				param.data[3] = -3;

	def get_init_states(self, batch_size, device):
		h_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		v_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		dU_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim).to(device);
		trace_e_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
		trace_E_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim).to(device);
		return h_0, v_0, dU_0, (trace_e_0, trace_E_0);

class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers=1, \
				  dropout_e = 0, dropout_i = 0, dropout_h = 0, dropout_o = 0, dropout_w = 0,\
          padding_idx=None, activation="swish", tie_weight=True):

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

		assert(in_type in ["continuous", "categorical", "image"]), "Please input the correct input type";
		assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
		assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

		if in_type=="categorical":
			self.encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
		elif in_type=="image":
			self.conv1 = torch.nn.Conv2d(1, 32, 4, 3); # 32@9*9
			self.conv2 = torch.nn.Conv2d(32, 32, 3, 2); # 32@4*4
			self.bn1 = torch.nn.BatchNorm2d(16);
			self.bn2 = torch.nn.BatchNorm2d(32);
			self.fc1 = nn.Linear(256, self.input_dim);
			def encode(x):
				x = self.bn1(torch.relu(self.conv1(x)));
				x = self.bn2(torch.relu(self.conv2(x)));
				x = torch.flatten(x, 1);
				x = torch.relu(self.fc1(x));
				return x;
			self.encoder = encode;

		self.rnns = [];
		self.rnns.append(SGRUCell(input_dim, hidden_dim, activation));

		if (self.tie_weight):
			for i in range(1, num_layers-1):
				self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation));
			self.rnns.append(SGRUCell(hidden_dim, input_dim, activation));
		else:
			for i in range(1, num_layers):
				self.rnns.append(SGRUCell(hidden_dim, hidden_dim, activation));

		if (self.dropout_w):
			self.rnns = [WeightDrop(self.rnns[l], ["h2h_weight", "h2mod_weight", "mod2h_weight"], self.dropout_w) for l in range(num_layers)];

		self.rnns = torch.nn.ModuleList(self.rnns);

		if out_type=="continuous":
			self.decoder = torch.nn.Linear(hidden_dim, out_dim);
		elif out_type=="categorical":
			if (self.tie_weight):
				self.decoder = torch.nn.Sequential(torch.nn.Linear(input_dim, out_dim), torch.nn.LogSoftmax(dim=2));
				self.decoder[0].weight = self.encoder.weight;
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

		if self.out_type=="continuous":
			torch.nn.init.xavier_normal_(self.decoder.weight.data);
			torch.nn.init.zeros_(self.decoder.bias.data);
		elif not self.tie_weight or self.out_type!="categorical":
			torch.nn.init.xavier_normal_(self.decoder[0].weight.data);
			torch.nn.init.zeros_(self.decoder[0].bias.data);