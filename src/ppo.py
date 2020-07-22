import torch, copy

eps = torch.finfo(torch.float32).eps
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
the Memory and PPO classes are taken from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py, adapting for recurrent policy
'''

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]

class PPO:
    def __init__(self, policy, optimizer, seq_len, buffer_size, beta_v = 0.5, beta_entropy = 0.001, gamma=0.99, lambd=0.95, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma;
        self.lambd = lambd;
        self.eps_clip = eps_clip;
        self.K_epochs = K_epochs;
        self.seq_len = seq_len;
        self.buffer_size = buffer_size;
        self.beta_v = beta_v;
        self.beta_entropy = beta_entropy;
        
        self.policy = policy;
        self.optimizer = optimizer;
        self.value_loss = torch.nn.MSELoss();
    
    def update(self, memory, task='cardsort'):   
        rewards = torch.as_tensor(memory.rewards).to(device).detach().t();
        values = torch.as_tensor(memory.values).to(device).detach().t();

        # calculate advantages
        returns = torch.zeros(self.seq_len, self.buffer_size).to(device);
        A = torch.zeros(self.buffer_size).to(device);
        v_next = torch.zeros(self.buffer_size).to(device);
        for idx in range(1, self.seq_len+1):
            td_error = rewards[-idx] + self.gamma*v_next - values[-idx];
            A = td_error + A * self.gamma * self.lambd
            v_next = values[-idx];
            returns[-idx] = A + values[-idx];

        # Normalizing the rewards:
        returns = (returns - returns.mean()) / (returns.std() + 1e-5);
        
        # convert list to tensor
        old_states = torch.stack([torch.stack(m) for m in memory.states]).to(device).detach().transpose(0,3).squeeze(0);
        old_actions = torch.as_tensor(memory.actions).to(device).detach().t()
        old_logprobs = torch.as_tensor(memory.logprobs).to(device).detach().t();
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            new_h, new_v, new_dU, new_trace = self.policy.get_init_states(batch_size=self.buffer_size, device=device);
            loss = 0;
            # for each timestep, give a batch of old observations
            for s, a, r, old_log_prob in zip(old_states, old_actions, returns, old_logprobs):
                new_v, new_h, new_dU, new_trace, (last_layer_out, log_probs, value), mod = self.policy.train().forward(\
                                                      x = s.to(device),\
                                                      h = new_h, \
                                                      v = new_v, \
                                                      dU = new_dU, \
                                                      trace = new_trace);
                m = torch.distributions.Categorical(logits = log_probs[-1]);
                # calculate entropy loss
                dist_entropy = m.entropy().mean();
                # calculate new policy, size = (batch,)
                logprobs = m.log_prob(a);
            
                # Finding the ratio (pi_theta / pi_theta__old), 
                ratios = torch.exp(logprobs - old_log_prob.detach());

                adv = r - value[-1].squeeze().detach();

                # Finding Surrogate Loss:
                surr1 = ratios * adv;
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv;
                loss += -torch.min(surr1, surr2).mean() + self.beta_v*self.value_loss(value[-1].squeeze(), r) - self.beta_entropy*dist_entropy;
                loss += 0.01*(last_layer_out.pow(2).mean());

            loss.backward()
            
            # take gradient step after all the episodes are evaluated
            torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), 1.0);
            self.optimizer.step();
            self.optimizer.zero_grad();