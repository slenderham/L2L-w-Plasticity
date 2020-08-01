# %%
import torch
from modulated_AC import SGRU
import torch.optim as optim
from scipy.stats import ortho_group
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb
from ppo import PPO, Memory
import tqdm

'''
    simpler WCST
    the episodes are predetermined, but whether the dimension to attend to changes from trial to trial
    however, it stays constant during each trial
    the task is to quickly

'''

torch.manual_seed(37);

def add_weight_decay(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if ("ln" in name or "encoder" in name or "weight" not in name):
            no_decay.append(param);
        else:
            decay.append(param);

    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 0.}];

def sample_card(chunks, bits, val):

    '''
        input:
            chunks: the number of dimensions (color, shape, numer of shapes...)
            bits: the number of variations along each dimensions
        output:
            randInts: the "card" with different feature along each dimension
            pattern: binary coded
    '''
    global data;

    randInts = torch.tensor(np.random.randint(0, val, size=(chunks,)));
    pattern = data[randInts, :];

    return randInts.to(device), pattern.to(device);


# how many dimensions
chunks = 3;

# the possible values each input dimension can take
val = 4;

# the size of each input dimension
bits = 4;

sampler = lambda : sample_card(chunks, bits, val);

model = SGRU(in_type = "continuous",\
             out_type = "categorical",\
             num_token = 0,\
             input_dim = bits+val+1,\
             hidden_dim = 64,\
             out_dim = val,\
             num_layers = 1,\
             activation="relu",\
             mod_rank = 16\
            );

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.to(device);
else:
    device = torch.device("cpu");

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=1e-3);
# optimizer = optim.SGD(param_groups, lr=1);
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6);

n_epochs = 50000;
len_seq = 80;
val_len = 80;
buffer_size = 50;
num_samples = 50;

# data = torch.tensor(ortho_group.rvs(bits), dtype=torch.float, device=device);
data = torch.eye(bits);

episode_buffer = Memory();
cumReward = [];

try:
  state_dict = torch.load("model_WCST");
  model.load_state_dict(state_dict["model_state_dict"]);#print(model.state_dict());
  optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
  cumReward = state_dict["cumReward"];
except:
    print("model failed to load");

ppo = PPO(policy = model, \
          optimizer = optimizer, \
          seq_len = len_seq,\
          buffer_size = buffer_size,\
         );

for i in tqdm.tqdm(range(n_epochs), position=0, leave=True):


    # initialize loss and reward
    reward = torch.zeros(1, 1, device=device);
    new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
    action = torch.zeros(1, val, device=device);

    instrInts, instrPattern = sampler();
    newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);

    episode_buffer.actions.append([]);
    episode_buffer.states.append([]);
    episode_buffer.logprobs.append([]);
    episode_buffer.rewards.append([]);
    episode_buffer.values.append([]);

    next_change = 20;

    with torch.no_grad():
        for idx in range(len_seq):
            instrInts, instrPattern = sampler();
            if (idx==next_change):
                newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);
                next_change += 20;

            # RNN works with size seq len X batch size X input size, in this case # chunks X 1 X pattern size + |A| + 1
            patterns = torch.cat(
                                    (instrPattern.reshape(chunks, 1, bits), torch.zeros((chunks, 1, val+1), device=device)), dim=2
                                );
            # feedback from previous trial, 1 X 1 X [0]*pattern_size + previous action + previous reward
            feedback = torch.cat(
                                    (torch.zeros((1, bits), device=device), action.detach(), reward.detach()), dim=1
                                ).reshape(1, 1, bits+val+1);

            total_input = torch.cat(
                                    (feedback, patterns, torch.zeros(1, 1, bits+val+1)), dim=0
                                );
            # one iter of network, notice that the reward is from the previous time step
            new_v, new_h, new_dU, new_trace, (last_layer_out, log_probs, value), _ = model.train().forward(\
                                                          x = total_input.to(device),\
                                                          h = new_h, \
                                                          v = new_v, \
                                                          dU = new_dU, \
                                                          trace = new_trace);

            # sample an action
            m = torch.distributions.Categorical(logits = log_probs[-1]);
            action_idx = m.sample();
            action = torch.zeros(1, val, dtype=torch.float, device=device);
            action[:, action_idx] = 1.0;

            # get reward
            reward = torch.as_tensor((instrInts[newDim].flatten()==action_idx), dtype=torch.float, device=device).reshape(1,-1).detach();

            episode_buffer.actions[-1].append(action_idx);
            episode_buffer.states[-1].append(total_input); # batch_size, trial number, within trial time step, 1, input_dim
            episode_buffer.logprobs[-1].append(m.log_prob(action_idx));
            episode_buffer.rewards[-1].append(reward);
            episode_buffer.values[-1].append(value[-1]);

    # update the policy every [buffer_size] steps
    if (i+1)%buffer_size==0:
        ppo.update(episode_buffer);
        cumReward.append(torch.mean(torch.as_tensor(episode_buffer.rewards)));
        print(cumReward[-1]);
        episode_buffer.clear_memory();
        # scheduler1.step();
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'cumReward': cumReward}, 'model_WCST');



state_dict = torch.load("model_WCST");
print(model.state_dict().keys())
print(state_dict["model_state_dict"].keys())
model.load_state_dict(state_dict["model_state_dict"]);


dUs = [];
vs = [];
dims = [];
cumReward = [];
with torch.no_grad():

    for j in range(num_samples):

        dUs.append([]);
        vs.append([]);
        dims.append([]);
        cumReward.append([]);

        # data = torch.as_tensor(torch.rand(val, bits)>0.5, dtype=torch.float, device=device);

        reward = torch.zeros(1, 1, device=device);
        new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
        action = torch.zeros(1, val, device=device);

        instrInts, instrPattern = sampler();
        newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);

        next_change = 20;

        for idx in range(len_seq):
            instrInts, instrPattern = sampler();
            if (idx==next_change):
                newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);
                next_change += 20;

            # RNN works with size seq len X batch size X input size, in this case # chunks X 1 X pattern size + |A| + 1
            patterns = torch.cat(
                  (instrPattern.reshape(chunks, 1, bits), torch.zeros((chunks, 1, val+1), device=device)), dim=2
                );
            # feedback from previous trial, 1 X 1 X [0]*pattern_size + previous action + previous reward
            feedback = torch.cat(
                  (torch.zeros((1, bits), device=device), action.detach(), reward.detach()), dim=1
                ).reshape(1, 1, bits+val+1);

            total_input = torch.cat(
                  (feedback, patterns), dim=0
                );
            # one iter of network, notice that the reward is from the previous time step
            for jdx in range(chunks+1):
                new_v, new_h, new_dU, new_trace, (last_layer_out, log_probs, value) = model.train().forward(\
                                  x = total_input[jdx:jdx+1,...].to(device),\
                                  h = new_h, \
                                  v = new_v, \
                                  dU = new_dU, \
                                  trace = new_trace);
                dUs[-1].append(new_dU[0]);
                vs[-1].append(new_v[0]);
                dims[-1].append(newDim);

            # sample an action
            m = torch.distributions.Categorical(logits = log_probs[-1]);
            action_idx = m.sample();
            action = torch.zeros(1, val, dtype=torch.float, device=device);
            action[:, action_idx] = 1.0;

            # get reward
            reward = torch.as_tensor((instrInts[newDim].flatten()==action_idx), dtype=torch.float, device=device).reshape(1,-1).detach();
            cumReward[-1].append(1-reward.item());

print(np.mean(cumReward));
plt.imshow(cumReward);

# %%
