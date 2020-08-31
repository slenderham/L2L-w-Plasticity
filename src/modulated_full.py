import torch
import math
from activations import SaturatingPoisson, Swish, Spike, TernaryTanh, kWTA, NormalizeWeight, sigma_inv
from dropouts import embedded_dropout, LockedDropout, WeightDrop


class SGRUCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, mod_rank, inits, clip_val=1.0, bias=True):

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
        self.inits = inits;

        # input-hidden weights
        self.x2h = torch.nn.Linear(input_dim, 2*hidden_dim, bias=bias);
        # hidden-hidden weights
        self.h2h = torch.nn.Linear(hidden_dim, 2*hidden_dim, bias=bias);

        self.lnx = torch.nn.LayerNorm(2*hidden_dim);
        self.lnh = torch.nn.LayerNorm(2*hidden_dim);

        self.h2mod = torch.nn.Linear(hidden_dim, mod_rank, bias=bias);

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
        self.alpha = torch.nn.Parameter(-4.0*torch.ones(1));
        self.mod2h = torch.nn.Linear(mod_rank, 3);

        # time constant of STDP weight modification
        self.tau_U = torch.nn.Parameter(-4.5*torch.ones(1));
        self.reset_parameter();

    def forward(self, x, h, v, dU, trace):
        curr_out = [];
        mods = [];
        keys = [];
        dicts = [];
        for c in range(x.shape[0]):
            v, h, dU, trace, mod = self._forward_step(x[c], h, v, dU, trace);
            curr_out.append(h);
            mods.append(mod);
            keys.append(trace[0]);
            dicts.append(dU);
        return v, h, dU, trace, torch.stack(curr_out), torch.stack(mods), torch.stack(keys), torch.stack(dicts);

    def _forward_step(self, x, h, v, dU, trace):
        trace_e, trace_E = trace;

        # Wx = self.x2h(x);
        # Wh = self.h2h(h);
        # Wh[:, 2*self.hidden_dim:3*self.hidden_dim] += torch.bmm(torch.nn.functional.softplus(self.alpha)*dU, h.unsqueeze(2)).squeeze(2);


        # preactivations
        Wx = self.lnx(self.x2h(x));
        Wh = self.h2h(h);
        Wh[:, 1*self.hidden_dim:2*self.hidden_dim] += torch.bmm(torch.nn.functional.softplus(self.alpha)*dU, h.unsqueeze(2)).squeeze(2);
        Wh = self.lnh(Wh);

        # segment into gates: forget and reset gate for GRU, concurrent STDP modulation for the eligibility trace
        # clip weight modification between for stability (only when it's used)

        z, dv = torch.split(Wx+Wh, [self.hidden_dim, self.hidden_dim], dim=-1);

        z = torch.sigmoid(z);
        v = (1-z) * v + z * dv;
        new_h = self.act(v);

        mod = self.mod2h(self.act(self.h2mod(new_h)));
        r, s, m = torch.split(mod, [1, 1, 1], dim=-1);
        r = torch.sigmoid(r);
        s = torch.sigmoid(s).unsqueeze(-1);
        m = (m).unsqueeze(-1);

        new_trace_e = (1-r)*trace_e + r*h;
        new_trace_E = (1-s)*trace_E + s*(torch.bmm(new_h.unsqueeze(2), new_trace_e.unsqueeze(1)) - torch.bmm(new_trace_e.unsqueeze(2), new_h.unsqueeze(1)));
        dU = (1-torch.sigmoid(self.tau_U))*dU+torch.sigmoid(self.tau_U)*m*new_trace_E;
        upper = torch.relu(self.clip_val-self.h2h.weight[2*self.hidden_dim:3*self.hidden_dim,:])/(torch.nn.functional.softplus(self.alpha)+1e-8);
        lower = -torch.relu(self.clip_val+self.h2h.weight[2*self.hidden_dim:3*self.hidden_dim,:])/(torch.nn.functional.softplus(self.alpha)+1e-8);
        dU = torch.where(dU>upper, upper, dU);
        dU = torch.where(dU<lower, lower, dU);
        
        return v, new_h, dU, (new_trace_e, new_trace_E), mod;

    def reset_parameter(self):
        for name, param in self.named_parameters():
            if "h2h.weight" in name:
                for i in range(2):
                    torch.nn.init.orthogonal_(param[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
            elif "x2h.weight" in name:
                for i in range(2):
                    torch.nn.init.xavier_normal_(param[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
            elif "x2h.bias" in name:
                torch.nn.init.zeros_(param);
            elif "h2h.bias" in name :
                torch.nn.init.zeros_(param);
                # param[:self.hidden_dim] = -1;
                # param[self.hidden_dim:2*self.hidden_dim] = -1;
            elif "h2mod.weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="relu");
            elif "h2mod.bias" in name:
                torch.nn.init.zeros_(param);
            elif "mod2h.weight" in name:
                for i in range(3):
                    torch.nn.init.kaiming_normal_(param[i:i+1,:], nonlinearity="relu");
            elif "mod2h.bias" in name:
                torch.nn.init.zeros_(param);
                # param[:self.hidden_dim] = -1;
                # param[self.hidden_dim:] = +1;
            # setattr(self, name.replace(".", "_"), param); 

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
                  alpha_init = 0.01, tau_U_init = -3.0, clip_val=1, reps=4,\
                  mod_rank = 1, padding_idx=None, activation="swish", tie_weight=True):

        super(SGRU, self).__init__();

        self.hidden_dim = hidden_dim;
        self.input_dim = input_dim;
        self.out_dim = out_dim;
        self.num_token = num_token;
        self.in_type = in_type;
        self.out_type = out_type;
        self.tie_weight = tie_weight if (in_type=="categorical" and out_type=="categorical") else False;
        self.reps = reps;

        self.dropout_e = dropout_e;
        self.dropout_i = dropout_e;
        self.dropout_h = dropout_h;
        self.dropout_o = dropout_o;
        self.dropout_w = dropout_w;

        assert(in_type in ["continuous", "categorical", "image", "image+categorical"]), "Please input the correct input type";
        assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
        assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

        if in_type=="categorical":
            self.encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
        elif in_type=="image" or in_type=="image+categorical":
            self.conv1 = torch.nn.Conv2d(1, 64, 3, 1, 1); # 64@28*28
            self.pool1 = torch.nn.MaxPool2d(2, 2); #64@14*14
            self.bn1 = torch.nn.BatchNorm2d(64);

            self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1); # 64@14*14
            self.pool2 = torch.nn.MaxPool2d(2, 2); #64@7*7
            self.bn2 = torch.nn.BatchNorm2d(64);

            self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1); # 64@7*7
            self.pool3 = torch.nn.MaxPool2d(2, 2); #64@3*3
            self.bn3 = torch.nn.BatchNorm2d(64);

            self.conv4 = torch.nn.Conv2d(64, 64, 3, 1, 1); # 64@3*3
            self.pool4 = torch.nn.MaxPool2d(2, 2); # 64@1*1
            self.bn4 = torch.nn.BatchNorm2d(64);

            def encode(x):
                x = self.bn1(torch.relu(self.pool1(self.conv1(x))));
                x = self.bn2(torch.relu(self.pool2(self.conv2(x))));
                x = self.bn3(torch.relu(self.pool3(self.conv3(x))));
                x = self.bn4(torch.relu(self.pool4(self.conv4(x))));
                x = torch.flatten(x, 1);
                return x;
            self.img_encoder = encode;
            self.label_encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
        
        if (self.in_type=="image+categorical"):
            input_dim = 64+self.num_token;
        
        inits = {"alpha_init": alpha_init, "tau_U_init": tau_U_init};

        self.rnns = [];
        self.rnns.append(SGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, activation=activation, mod_rank=mod_rank, inits=inits, clip_val=clip_val));

        if (self.tie_weight):
            for i in range(1, num_layers-1):
                self.rnns.append(SGRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation, mod_rank=mod_rank, inits=init, sclip_val=clip_val));
            self.rnns.append(SGRUCell(input_dim=hidden_dim, hidden_dim=input_dim, activation=activation, mod_rank=mod_rank, inits=inits, clip_val=clip_val));
        else:
            for i in range(1, num_layers):
                self.rnns.append(SGRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation, mod_rank=mod_rank, inits=inits, clip_val=clip_val));

        if (self.dropout_w):
            self.rnns = [WeightDrop(self.rnns[l], ["h2h_weight"], self.dropout_w) for l in range(num_layers)];

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

    def forward(self, x, h, v, dU, trace, **kwargs):

        # size of x is seq_len X batch_size X input_dimension

        if self.in_type=="categorical":
            new_x = embedded_dropout(self.encoder, x, dropout=self.dropout_e if self.training else 0);
        elif self.in_type=="image+categorical":
            time, batch_size, channel, height, width = x[0].shape;
            img = self.img_encoder(x[0].reshape(time*batch_size, channel, height, width)).reshape(time, batch_size, 64);
            lbl = (torch.eye(self.num_token)[x[1]]*math.sqrt(64)).to(img.device)
            new_x = torch.cat([img, lbl], dim=-1);
            new_x = torch.repeat_interleave(new_x, self.reps, dim=0);
            new_x = torch.nn.Parameter(new_x, requires_grad=True);

        prev_out = self.locked_drop(new_x, self.dropout_i);

        multi_mods = [];

        for l, rnn in enumerate(self.rnns):
            v[l], h[l], dU[l], trace[l], prev_out, mods, keys, dicts = rnn.forward(prev_out, h[l], v[l], dU[l], trace[l]);
            multi_mods.append(mods);

            if l!=self.num_layers-1:
                prev_out = self.locked_drop(prev_out, self.dropout_h);
            else:
                prev_out = self.locked_drop(prev_out, self.dropout_o);

        return v, h, dU, trace, ({'vals':prev_out, 'keys':keys, 'dicts':dicts, 'new_x':new_x}, self.decoder(prev_out));

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
        for n, p in self.named_parameters():
            print(n, p.numel());
        print(sum([p.numel() for p in self.parameters()]));

        if self.in_type=="categorical":
            torch.nn.init.xavier_uniform_(self.encoder.weight);
        elif self.in_type=="image+categorical":
            torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu");
            torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu");
            torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu");
            torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu");
            torch.nn.init.zeros_(self.conv1.bias);
            torch.nn.init.zeros_(self.conv2.bias);
            torch.nn.init.zeros_(self.conv3.bias);
            torch.nn.init.zeros_(self.conv4.bias);
      
        if self.out_type=="continuous":
            torch.nn.init.xavier_normal_(self.decoder.weight);
            torch.nn.init.zeros_(self.decoder.bias);
        elif not self.tie_weight or self.out_type!="categorical":
            torch.nn.init.xavier_normal_(self.decoder[0].weight);
            torch.nn.init.zeros_(self.decoder[0].bias);