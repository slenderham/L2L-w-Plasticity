import torch
import math
from activations import SaturatingPoisson, Swish, Spike, TernaryTanh, kWTA, NormalizeWeight
from dropouts import embedded_dropout, LockedDropout, WeightDrop


class SGRUCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, mod_rank, inits=None, clip_val=1.0, bias=True):

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
        self.x2h = torch.nn.Linear(input_dim, 3*hidden_dim, bias=bias);
        # hidden-hidden weights
        self.h2h = torch.nn.Linear(hidden_dim, 3*hidden_dim, bias=bias);

        self.lnx = torch.nn.LayerNorm(3*hidden_dim);
        self.lnh = torch.nn.LayerNorm(3*hidden_dim);

        self.h2mod = torch.nn.Linear(hidden_dim, 3*mod_rank, bias=bias);

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
        self.alpha = torch.nn.Parameter(-4.5*torch.ones(1));
        self.mod2hr = torch.nn.Linear(mod_rank, 1);
        self.mod2hs = torch.nn.Linear(mod_rank, 1);
        self.mod2hm = torch.nn.Linear(mod_rank, 1);

        # time constant of STDP weight modification
        self.tau_U = torch.nn.Parameter(-4.5*torch.ones(1));
        self.reset_parameter();

    def forward(self, x, h, v, dU, trace, **kwargs):
        freeze_fw = kwargs.get("freeze_fw")
        curr_out = [];
        mods = [];
        ss = [];
        ms = [];
        rs = []
        dUs = []
        for c in range(x.shape[0]):
            v, h, dU, trace, (mod, s, m, r) = self._forward_step(x[c], h, v, dU, trace, freeze_fw=freeze_fw[c] if freeze_fw is not None else False);
            curr_out.append(h);
            mods.append(mod);
            ss.append(s);
            ms.append(m);
            rs.append(r);
            dUs.append(dU);
        return v, h, dU, trace, torch.stack(curr_out), torch.stack(dUs), (torch.stack(mods), torch.stack(ss), torch.stack(ms), torch.stack(rs));

    def _forward_step(self, x, h, v, dU, trace, **kwargs):
        freeze_fw = kwargs.get("freeze_fw")
        trace_e, trace_E = trace;

        # Wx = self.x2h(x);
        # Wh = self.h2h(h);
        # Wh[:, self.hidden_dim:2*self.hidden_dim] += torch.bmm(torch.nn.functional.softplus(self.alpha)*dU, h.unsqueeze(2)).squeeze(2);

        # preactivations
        Wx = self.lnx(self.x2h(x));
        Wh = self.h2h(h);
        Wh[:, -self.hidden_dim:] += torch.bmm(torch.nn.functional.softplus(self.alpha)*dU, h.unsqueeze(2)).squeeze(2);
        Wh = self.lnh(Wh);

        # segment into gates: forget and reset gate for GRU, concurrent STDP modulation for the eligibility trace
        # clip weight modification between for stability (only when it's used)

        z, o, dv = torch.split(Wx+Wh, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1);

        o = torch.sigmoid(o);
        z = torch.sigmoid(z);
        v = (1-z) * v + z * dv;
        new_h = self.act(v);

        mod = self.act(self.h2mod(new_h));
        r, s, m = torch.split(mod, [self.mod_rank, self.mod_rank, self.mod_rank], dim=-1);
        r = torch.sigmoid(self.mod2hr(r));
        s = torch.sigmoid(self.mod2hs(s)).unsqueeze(-1);
        m = torch.nn.functional.tanhshrink(self.mod2hm(m)).unsqueeze(-1);

        if not freeze_fw:
            h_for_fw = o*new_h;
            new_trace_e = (1-r)*trace_e + r*h_for_fw;
            new_trace_E = (1-s)*trace_E + s*(torch.bmm(h_for_fw.unsqueeze(2), trace_e.unsqueeze(1)) - torch.bmm(trace_e.unsqueeze(2), h_for_fw.unsqueeze(1)));
            dU = (1-torch.sigmoid(self.tau_U))*dU+torch.sigmoid(self.tau_U)*m*new_trace_E;
            upper = torch.relu(self.clip_val-self.h2h.weight[-self.hidden_dim:,:])/(torch.nn.functional.softplus(self.alpha)+1e-8);
            lower = -torch.relu(self.clip_val+self.h2h.weight[-self.hidden_dim:,:])/(torch.nn.functional.softplus(self.alpha)+1e-8);
            dU = torch.where(dU>upper, upper, dU);
            dU = torch.where(dU<lower, lower, dU);
        else:
            new_trace_E = trace_E
            new_trace_e = trace_e
        
        return v, new_h, dU, (new_trace_e, new_trace_E), (mod, s, m, r);

    def reset_parameter(self):
        for name, param in self.named_parameters():
            if "h2h.weight" in name:
                for i in range(3):
                    torch.nn.init.orthogonal_(param[i*self.hidden_dim:(i+1)*self.hidden_dim,:], gain=math.sqrt(2));
            elif "x2h.weight" in name:
                for i in range(3):
                    torch.nn.init.kaiming_normal_(param[i*self.hidden_dim:(i+1)*self.hidden_dim,:]);
            elif "x2h.bias" in name:
                torch.nn.init.zeros_(param);
            elif "h2h.bias" in name :
                torch.nn.init.zeros_(param);
                # param[:self.hidden_dim] = -1;
                # param[self.hidden_dim:2*self.hidden_dim] = -1;
            elif "h2mod.weight" in name:
                for i in range(3):
                  torch.nn.init.kaiming_normal_(param[i*self.mod_rank:(i+1)*self.mod_rank,:]);
            elif "h2mod.bias" in name:
                torch.nn.init.zeros_(param);
            elif "mod2h" in name and "weight" in name:
                torch.nn.init.kaiming_normal_(param);
            elif "mod2h" in name and "bias" in name:
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
                  mod_rank = 1, padding_idx=None, activation="swish", tie_weight=True):

        super(SGRU, self).__init__();

        self.hidden_dim = hidden_dim;
        self.input_dim = input_dim;
        self.out_dim = out_dim;
        self.num_token = num_token;
        self.in_type = in_type;
        self.out_type = out_type;

        assert(in_type in ["continuous", "categorical", "image+continuous"]), "Please input the correct input type";
        assert(out_type in ["continuous", "categorical", "binary"]), "Please input the correct output type";
        assert(activation in ["relu", "softplus", "tanh", "poisson", "swish"]), "please use a correct activation function";

        if in_type=="categorical":
            self.encoder = torch.nn.Embedding(num_token, input_dim, padding_idx=padding_idx);
        elif in_type=="image+continuous":
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
            self.encoder = encode;
            input_dim += 64;

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
            self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.LogSigmoid());

        self.num_layers = num_layers;
        self.critic = torch.nn.Linear(hidden_dim*num_layers, 1);

        # dropouts
        self.reset_parameter();

    def forward(self, x, h, v, dU, trace, **kwargs):

        # size of x is seq_len X batch_size X input_dimension

        if self.in_type=="categorical":
            prev_out = self.encoder(x);
        elif self.in_type=="image+continuous":
            num_steps, batch_size = x[0].shape[:2]
            img = self.encoder(x[0].flatten(0, 1)).reshape(num_steps, batch_size, -1);
            prev_out = torch.cat([img, x[1]], dim=-1);
        else:
            prev_out = x;

        for l, rnn in enumerate(self.rnns):
            v[l], h[l], dU[l], trace[l], prev_out, fws, mod = rnn.forward(prev_out, h[l], v[l], dU[l], trace[l], **kwargs);

        action_logit = self.decoder(prev_out);
        value = self.critic(prev_out);

        return v, h, dU, trace, (prev_out, fws, action_logit, value), mod;

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
        for n,p in self.named_parameters():
          print(n, p.numel());
        print(sum([p.numel() for p in self.parameters()]));

        if self.in_type=="categorical":
            torch.nn.init.xavier_uniform_(self.encoder.weight);
            
        if self.out_type=="categorical" or self.out_type=="binary":
            torch.nn.init.xavier_normal_(self.decoder[0].weight);
            torch.nn.init.zeros_(self.decoder[0].bias);
        elif self.out_type=="continuous":
            torch.nn.init.xavier_normal_(self.decoder.weight);
            torch.nn.init.zeros_(self.decoder.bias);