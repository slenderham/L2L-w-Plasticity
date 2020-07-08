import torch
import math
from activations import SaturatingPoisson, Swish, Spike, TernaryTanh, kWTA, NormalizeWeight, sigma_inv
from dropouts import embedded_dropout, LockedDropout, WeightDrop

class SGRUCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, mod_rank, inits, bias=True):

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

        self.x2h = torch.nn.Linear(input_dim, 3*hidden_dim, bias=bias);
        self.h2h = torch.nn.Linear(hidden_dim, 3*hidden_dim, bias=bias);
        self.att2h = torch.nn.Linear(mod_rank, 3*hidden_dim, bias=bias);

        self.lnx = torch.nn.LayerNorm(3*hidden_dim);
        self.lnh = torch.nn.LayerNorm(3*hidden_dim);
        self.lnatt = torch.nn.LayerNorm(3*hidden_dim);

        self.h2att = torch.nn.Linear(hidden_dim, mod_rank+3, bias=bias);
        self.att2h = torch.nn.Linear(mod_rank, hidden_dim, bias=bias);

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
        
        self.att_act = kWTA(sparsity=0.1);

        # strength of weight modification
        self.mod2h = torch.nn.Linear(mod_rank, 2 * hidden_dim);

        # time constant of STDP weight modification
        self.lambd = torch.nn.Parameter(-3.0*torch.ones(1));
        self.eta = torch.nn.Parameter(3.0*torch.ones(1));
        self.tau_att = torch.nn.Parameter(-2.0*torch.ones(1));
        self.reset_parameter();

    def forward(self, x, states):
        curr_out = [];
        mods = [];
        for c in range(x.shape[0]):
            states = self._forward_step(x[c], states);
            curr_out.append(states["h"]);
            mods.append(states["mod"]);
        return states, torch.stack(curr_out), torch.stack(mods);

    def _forward_step(self, x, states):
        h, v, dU, trace_e, trace_E, h_att = states["h"], states["v"], states["dU"], states["trace_e"], states["trace_E"], states["h_att"];

        # Wx = self.x2h(x);
        # Wh = self.h2h(h);

        # preactivations
        Wx = self.lnx(self.x2h(x));
        Wh = self.lnh(self.h2h(h));
        Watt = self.lnatt(self.att2h(h_att));

        # segment into gates: forget and reset gate for GRU
        zx, rx, dvx = torch.split(Wx, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1);
        zh, rh, dvh = torch.split(Wh, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1);
        zatt, ratt, dvatt = torch.split(Watt, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1);

        z = torch.sigmoid(zx+zh+zatt);
        r = torch.sigmoid(rx+rh+ratt);
        v = (1-z) * v + z * r * (dvx+dvh+dvatt);
        new_h = self.act(v);

        # project h to memory matrix
        mod = self.act(self.h2mod(new_h));

        # for the attractor network
        # d: decay of memory trace
        # s: decay for 
        d, s, m, att_in = torch.split(mod, [1, 1, 1, self.mod_rank], dim=-1);
        
        s = torch.sigmoid(s).unsqueeze(-1);
        d = torch.sigmoid(d).unsqueeze(-1);
        m = torch.tanh(m).unsqueeze(-1);
        h_att_new = (1-torch.sigmoid(self.tau_att))*h_att + torch.sigmoid(self.tau_att)*self.att_act(h2att + torch.bmm(dU, h_att.unsqueeze(2)).squeeze(2));

        new_trace_e = (1-d)*trace_e + d*h_att;
        new_trace_E = (1-s)*trace_E + s*(torch.bmm(h_att_new.unsqueeze(2), new_trace_e.unsqueeze(1)) - torch.bmm(new_trace_e.unsqueeze(2), h_att_new.unsqueeze(1)));
        dU = (1-torch.sigmoid(self.lambd))*dU+torch.sigmoid(self.eta)*m*new_trace_E;
        
        return {"h": new_h, "v": v, "dU": dU, "trace_e": new_trace_e, "trace_E": new_trace_E, "h_att": h_att_new};

    def reset_parameter(self):
        for name, param in self.named_parameters():
            if "h2h.weight" in name:
                for i in range(2):
                    torch.nn.init.xavier_normal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
                torch.nn.init.orthogonal_(param.data[2*self.hidden_dim:3*self.hidden_dim,:], gain=math.sqrt(2));
            elif "x2h.weight" in name:
                for i in range(2):
                    torch.nn.init.xavier_normal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
                torch.nn.init.kaiming_normal_(param.data[2*self.hidden_dim:3*self.hidden_dim,:], nonlinearity="relu");
            elif "x2h.bias" in name:
                torch.nn.init.zeros_(param.data);
            elif "h2h.bias" in name :
                torch.nn.init.zeros_(param.data);
                # param.data[:self.hidden_dim] = -1;
                # param.data[self.hidden_dim:2*self.hidden_dim] = -1;
            elif "h2mod.weight" in name:
                torch.nn.init.kaiming_normal_(param.data, nonlinearity="relu");
            elif "h2mod.bias" in name:
                torch.nn.init.zeros_(param.data);
            elif "mod2h.weight" in name:
                for i in range(2):
                    torch.nn.init.xavier_normal_(param.data[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
            elif "mod2h.bias" in name:
                torch.nn.init.zeros_(param.data);
                # param.data[:self.hidden_dim] = -1;
                # param.data[self.hidden_dim:] = +1;
            # setattr(self, name.replace(".", "_"), param); 

    def get_init_states(self, batch_size, device):
        h_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
        v_0 = torch.zeros(batch_size, self.hidden_dim).to(device);
        dU_0 = torch.zeros(batch_size, self.mod_rank, self.mod_rank).to(device);
        trace_e_0 = torch.zeros(batch_size, self.mod_rank).to(device);
        trace_E_0 = torch.zeros(batch_size, self.mod_rank, self.mod_rank).to(device);
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
            # self.gn1 = torch.nn.GroupNorm(4, 64);

            self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1); # 64@14*14
            self.pool2 = torch.nn.MaxPool2d(2, 2); #64@7*7
            self.bn2 = torch.nn.BatchNorm2d(64);
            # self.gn2 = torch.nn.GroupNorm(4, 64);

            self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1); # 64@7*7
            self.pool3 = torch.nn.MaxPool2d(2, 2); #64@3*3
            self.bn3 = torch.nn.BatchNorm2d(64);
            # self.gn3 = torch.nn.GroupNorm(4, 64);

            self.conv4 = torch.nn.Conv2d(64, 64, 3, 1, 1); # 64@3*3
            self.pool4 = torch.nn.MaxPool2d(2, 2); # 64@1*1
            self.bn4 = torch.nn.BatchNorm2d(64);
            # self.gn4 = torch.nn.GroupNorm(4, 64);

            def encode(x):
                x = self.bn1(torch.relu(self.pool1(self.conv1(x))));
                x = self.bn2(torch.relu(self.pool2(self.conv2(x))));
                x = self.bn3(torch.relu(self.pool3(self.conv3(x))));
                x = self.bn4(torch.relu(self.pool4(self.conv4(x))));
                # x = self.gn1(torch.relu(self.pool1(self.conv1(x))));
                # x = self.gn2(torch.relu(self.pool2(self.conv2(x))));
                # x = self.gn3(torch.relu(self.pool3(self.conv3(x))));
                # x = self.gn4(torch.relu(self.pool4(self.conv4(x))));
                x = torch.flatten(x, 1);
                return x;
            self.img_encoder = encode;
            self.label_encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
        
        if (self.in_type=="image+categorical"):
            input_dim += 64;
        
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

    def forward(self, x, h, v, dU, trace):

        # size of x is seq_len X batch_size X input_dimension

        if self.in_type=="categorical":
            x = embedded_dropout(self.encoder, x, dropout=self.dropout_e if self.training else 0);
        elif self.in_type=="image+categorical":
            time, batch_size, channel, height, width = x[0].shape;
            img = self.img_encoder(x[0].reshape(time*batch_size, channel, height, width)).reshape(time, batch_size, 64);
            lbl = self.label_encoder(x[1]);
            x = torch.cat([img, lbl], dim=-1);
            x = torch.repeat_interleave(x, self.reps, dim=0);

        prev_out = self.locked_drop(x, self.dropout_i);

        multi_mods = [];

        for l, rnn in enumerate(self.rnns):
            v[l], h[l], dU[l], trace[l], prev_out, mods = rnn.forward(prev_out, h[l], v[l], dU[l], trace[l]);
            multi_mods.append(mods);

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
        for n, p in self.named_parameters():
            print(n, p.numel());
        print(sum([p.numel() for p in self.parameters()]));

        if self.in_type=="categorical":
            torch.nn.init.xavier_uniform_(self.encoder.weight.data);
      
        if self.out_type=="continuous":
            torch.nn.init.xavier_normal_(self.decoder.weight.data);
            torch.nn.init.zeros_(self.decoder.bias.data);
        elif not self.tie_weight or self.out_type!="categorical":
            torch.nn.init.xavier_normal_(self.decoder[0].weight.data);
            torch.nn.init.zeros_(self.decoder[0].bias.data);