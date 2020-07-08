import torch
import math
from utils.activations import SaturatingPoisson, Swish, Spike, LeakyIntegrate
from utils.dropouts import embedded_dropout, LockedDropout, WeightDrop

'''
a version of the LSNN by Maass
coming back to this later

'''


class SGRUCell(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, activation, mod_rank, bias=True):

		super(SGRUCell, self).__init__();

		self.input_dim = input_dim;
		self.hidden_dim = hidden_dim;
		self.bias = bias;

		# input-hidden weights
		self.x2h = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
		# hidden-hidden weights
		self.h2h = torch.nn.Linear(hidden_dim, hidden_dim, bias=bias)
		# layer norm
		self.lnx = torch.nn.LayerNorm(4*hidden_dim);
		self.lnh = torch.nn.LayerNorm(4*hidden_dim);

		# spiking function

		self.act = Spike();
		# strength of weight modification
		self.alpha = torch.nn.Parameter(torch.rand(1, hidden_dim)*math.sqrt(1/self.hidden_dim));
#		self.alpha = 0.01*torch.ones(1,1);

		# weight modification
		self.mod = torch.nn.Sequential(
										torch.nn.Linear(self.hidden_dim, mod_rank),
										torch.nn.LayerNorm(mod_rank, elementwise_affine=False),
										torch.nn.ReLU(),
										torch.nn.Linear(mod_rank, self.hidden_dim)
										);

		# Everything!!! (voltage, threshold adaption time constant, filtered spike train, eligibility trace, weight modification, threshold adaption magnitude)
		self.tau = torch.nn.Parameter(2*torch.rand(1, (5) * hidden_dim, 1)-1);

	def forward(self, x, trace):

		h, v, dU, trace_e, trace_E, threshold = trace;

		tau_v, tau_b, tau_e, tau_E, tau_U, tau_B = torch.chunk(self.tau, 5, dim=1);

		# preactivations
		Wx = self.lnx(self.x2h(x));
		Wh = self.lnh(self.h2h(h));

		# calculate input current: input, recurrent, plastic
		dv = Wx + Wh + torch.bmm(torch.abs(self.alpha)*torch.clamp(dU, -2, +2), h.unsqueeze(2)).squeeze(2);

		# integrate membrane voltage 
		new_v = (1-torch.sigmoid(tau_v)) * v + torch.sigmoid(tau_v) * dv;

		new_h = self.act(new_v, threshold);

		# reset membrane voltage
		new_v -= (new_h>0)*threshold;

		# new threshold
		new_thresh = (1-torch.sigmoid(tau_b))*threshold + torch.sigmoid(tau_b)*torch.abs(tau_B)*new_h;

		# calculate modulation
		m = self.mod(new_h);

		# calculate new filtered spike train
		new_trace_e = (1-torch.sigmoid(tau_e))*trace_e + torch.sigmoid(tau_e)*new_h;

		new_trace_E = (1-torch.sigmoid(tau_E))*trace_E + torch.sigmoid(tau_E)*(\
			torch.bmm((new_h.unsqueeze(2)), trace_e.unsqueeze(1)) - torch.bmm((trace_e.unsqueeze(2)), new_h.unsqueeze(1)));

		new_dU = (1-torch.sigmoid(self.tau_U))*dU+torch.sigmoid(self.tau_U)*m.unsqueeze(2)*new_trace_E;

		return (new_v, new_h, new_dU, new_trace_e, new_trace_E);

	def get_init_states(self, batch_size=1):
		h_0 = torch.zeros(batch_size, self.hidden_dim);
		v_0 = torch.zeros(batch_size, self.hidden_dim);
		dU_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim);
		trace_e_0 = torch.zeros(batch_size, self.hidden_dim);
		trace_E_0 = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim);
		return h_0, v_0, dU_0, (trace_e_0, trace_E_0);

class SGRU(torch.nn.Module):
	def __init__(self, in_type, out_type, num_token, input_dim, hidden_dim, out_dim, num_layers=1, \
				  dropout_e = 0, dropout_i = 0, dropout_h = 0, dropout_o = 0, \
				  mod_rank = 1, padding_idx=None, activation="swish"):
		super(SGRU, self).__init__();

		self.hidden_dim = hidden_dim;
		self.input_dim = input_dim;
		self.out_dim = out_dim;
		self.num_token = num_token;

		self.dropout_e = dropout_e;
		self.dropout_i = dropout_e;
		self.dropout_h = dropout_h;
		self.dropout_o = dropout_o;


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
			self.decoder = torch.nn.LeakyIntegrate(hidden_dim, out_dim);
		elif out_type=="categorical":
			self.decoder = torch.nn.Sequential(torch.nn.LeakyIntegrate(hidden_dim, out_dim), torch.nn.LogSoftmax(dim=2));
		else:
			self.decoder = torch.nn.Sequential(torch.nn.LeakyIntegrate(hidden_dim, out_dim), torch.nn.LogSigmoid(dim=2));

		# makes a stochastic decision: if GO, stop processing; else, make another recurrent layer
		self.GoNoGo = torch.nn.Sequential(
										torch.nn.Linear(self.hidden_dim, 1);
										torch.nn.Sigmoid();
										);

		self.value = torch.nn.Linear(self.hidden_dim, 1);

		self.num_layers = num_layers;

		# dropouts
		self.locked_drop = LockedDropout();

		self.reset_parameter();

	def forward(self, x, h, v, dU, trace):

		# size of x is seq_len X batch_size X input_dimension

		if self.encoder!=None:
			x = embedded_dropout(self.encoder, x, dropout=self.dropout_e if self.training else 0);

		prev_out = self.locked_drop(x, self.dropout_i);

		new_vs = [];
		new_hs = [];
		new_dUs = [];
		new_traces = [];

		output = [];


		for l, rnn in enumerate(self.rnns):
			new_v, new_h, new_dU, new_trace = v[l], h[l], dU[l], trace[l];
			curr_out = [];
			for c in range(prev_out.shape[0]):
				new_v, new_h, new_dU, new_trace = rnn.forward(prev_out[c], new_h, new_v, new_dU, new_trace);
				curr_out.append(new_h);

			new_vs.append(new_v);
			new_hs.append(new_h);
			new_dUs.append(new_dU);
			new_traces.append(new_trace);

			prev_out = torch.stack(curr_out);

			if l!=self.num_layers-1:
				prev_out = self.locked_drop(prev_out, self.dropout_h);
			else:
				prev_out = self.locked_drop(prev_out, self.dropout_o);


# 		for c in range(prev_out.shape[0]):
# 			output.append(self.decoder(prev_out[c]));

# 		output = torch.stack(output);
		output = self.decoder(prev_out);
		output = output.view(output.size(0)*output.size(1), output.size(2));


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
				for i in range(4):
					torch.nn.init.orthogonal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
				#  torch.nn.init.normal_(param.data, mean=0, std=1/self.hidden_dim**0.5);
			elif "x2h.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);
			elif "h2h.bias" in name :
				torch.nn.init.zeros_(param.data);
			elif "mod" in name and "weight" in name:
				torch.nn.init.xavier_normal_(param.data);
			elif "mod" in name and "bias" in name:
				torch.nn.init.zeros_(param.data);
			elif "encoder.weight" in name:
				torch.nn.init.xavier_uniform_(param.data);