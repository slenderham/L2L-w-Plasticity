# %%
import torch
from modulated_AC import SGRU
import torch.optim as optim
from scipy.stats import ortho_group, ttest_rel
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from ppo import PPO, Memory
import tqdm
from decomposition import *
# %matplotlib qt
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

    randInts = torch.from_numpy(np.random.randint(0, val, size=(chunks,)));
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
             mod_rank = 32,\
            );

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.to(device);
else:
    device = torch.device("cpu");

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=1e-3);
# scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6);

n_epochs = 0;
len_seq = 120;
buffer_size = 50;
num_samples = 5;

# data = torch.tensor(ortho_group.rvs(bits), dtype=torch.float, device=device);
data = torch.eye(bits);

episode_buffer = Memory();
cumReward = [];

try:
    state_dict = torch.load("pretrained_models/model_WCST");
    model.load_state_dict(state_dict["model_state_dict"]);#print(model.state_dict());
    optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
    cumReward = state_dict["cumReward"];
except:
    print("model failed to load");

ppo = PPO(policy = model, \
          optimizer = optimizer, \
          seq_len = len_seq,\
          buffer_size = buffer_size,\
          gamma=0.1, lambd=0.1,\
         );

# %% Model Training

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

    next_change = 15;

    with torch.no_grad():
        for idx in range(len_seq):
            instrInts, instrPattern = sampler();
            if (idx==next_change):
                newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);
                next_change += 15;

            # RNN works with size seq len X batch size X input size, in this case # chunks X 1 X pattern size + |A| + 1
            patterns = torch.cat(
                                    (instrPattern.reshape(chunks, 1, bits), torch.zeros((chunks, 1, val+1), device=device)), dim=2
                                );
            # feedback from previous trial, 1 X 1 X [0]*pattern_size + previous action + previous reward
            feedback = torch.cat(
                                    (torch.zeros((1, bits), device=device), action.detach(), reward.detach()), dim=1
                                ).reshape(1, 1, bits+val+1);

            total_input = torch.cat(
                                    (feedback, patterns, torch.zeros(1, 1, bits+val+1, device=device)), dim=0
                                );
            # one iter of network, notice that the reward is from the previous time step
            new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fws, log_probs, value), _ = model.train().forward(\
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
        cumReward.append(torch.mean(torch.as_tensor(episode_buffer.rewards)));
        print(cumReward[-1]);
        ppo.update(episode_buffer);
        episode_buffer.clear_memory();
        # scheduler1.step();
        torch.save({'model_state_dict': model.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict(), \
                    'cumReward': cumReward}, 'pretrained_models/model_WCST');


# %% Load Model and Freeze Weight Analysis
state_dict = torch.load("pretrained_models/model_WCST");
print(model.state_dict().keys())
print(state_dict["model_state_dict"].keys())
print(model.load_state_dict(state_dict["model_state_dict"]));
# model.rnns[0].alpha = torch.nn.Parameter(-torch.ones(1)*1e6)


freeze_fw_starts = list(range(0, len_seq, 5))+[10000]
freeze_fw_res = [];
freeze_fw_starts = [10000]

for freeze_fw_start in freeze_fw_starts:
    dUs = [];
    hs = [];
    dims = [];
    cumReward = [];
    actions = [];
    inputs = [];
    mod_ms = [];
    mod_ss = [];
    mod_rs = [];
    with torch.no_grad():
        for j in tqdm.tqdm(range(num_samples)):
            dUs.append([]);
            hs.append([]);
            dims.append([]);
            actions.append([]);
            cumReward.append([]);
            inputs.append([]);
            mod_ms.append([])
            mod_ss.append([])
            mod_rs.append([])

            # data = torch.as_tensor(torch.rand(val, bits)>0.5, dtype=torch.float, device=device);

            reward = torch.zeros(1, 1, device=device);
            new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
            action = torch.zeros(1, val, device=device);

            instrInts, instrPattern = sampler();
            newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);

            next_change = 15;

            for idx in range(len_seq):
                instrInts, instrPattern = sampler();
                if (idx==next_change):
                    newDim = torch.randint(low=0, high=chunks, size=(1,)).to(device);
                    next_change += 15;

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
                total_input.requires_grad = True;
                # one iter of network, notice that the reward is from the previous time step
                for inp in total_input:
                    new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fws, log_probs, value), (mod, ss, ms, rs) = model.train().forward(\
                                        x = inp.unsqueeze(0).to(device),\
                                        h = new_h, \
                                        v = new_v, \
                                        dU = new_dU, \
                                        trace = new_trace,
                                        freeze_fw = [idx>freeze_fw_start]*5);
                    dUs[-1].append(new_dU[0]);
                    hs[-1].append(new_h[0]);
                    dims[-1].append(newDim);
                    inputs[-1].append(inp);
                    mod_ms[-1].append(ms)
                    mod_ss[-1].append(ss)
                    mod_rs[-1].append(rs)

                # sample an action
                m = torch.distributions.Categorical(logits = log_probs[-1]);
                action_idx = m.sample();
                action = torch.zeros(1, val, dtype=torch.float, device=device);
                action[:, action_idx] = 1.0;
                actions[-1].extend([action_idx]*(chunks+2));

                # get reward
                reward = torch.as_tensor((instrInts[newDim].flatten()==action_idx), dtype=torch.float, device=device).reshape(1,-1).detach();
                cumReward[-1].append(1-reward.item());
            
        freeze_fw_res.append(np.mean(cumReward, axis=1));
        print(freeze_fw_res[-1].shape, freeze_fw_res[-1].mean(), freeze_fw_res[-1].std())

freeze_fw_res = np.array(freeze_fw_res)

t_res = []
for i in range(len(freeze_fw_starts)-1):
    t_res.append(ttest_rel(freeze_fw_res[i], freeze_fw_res[i+1])[1])

# %% Plot Error Rate by Weight Freezing Time

plt.bar(range(len(freeze_fw_res)), freeze_fw_res.mean(1))
plt.errorbar(range(len(freeze_fw_res)), freeze_fw_res.mean(1), 1.96*freeze_fw_res.std(1)/num_samples**0.5, fmt='.')
for i in range(len(freeze_fw_starts)-1):
    y = freeze_fw_res.mean(1)[i:i+2].max()+0.05
    plt.plot([i+0.1, i+0.1, i+0.9, i+0.9], [y, y+0.005, y+0.005, y], lw=1.5, c='k')
    plt.text(i+0.25, y+0.01, sig2asterisk(t_res[i]))
plt.xlabel('Start of Weight Freezing')
plt.xticks(range(len(freeze_fw_res)), labels=list(freeze_fw_starts[:-1])+['No Freeze'])
plt.ylabel('Error Rate')
plt.title('Error Rate with Different Weight Freezing Times')

# %% Plot Intra-episode Reward by Timestep

print(np.mean(cumReward));
mean_rwd = np.mean(cumReward, axis=0);
std_rwd = np.std(cumReward, axis=0)/num_samples**0.5;
plt.plot(np.arange(len_seq), mean_rwd);
plt.fill_between(np.arange(len_seq), mean_rwd-std_rwd, mean_rwd+std_rwd, alpha=0.2)
# plt.vlines(freeze_fw_start, ymin=-0.05, ymax=(mean_rwd+std_rwd).max()+0.05, linestyles='dashed', alpha=0.5)
# plt.imshow(cumReward)
plt.xlabel('Intra-Episode Timestep')
plt.ylabel('Episodes')
plt.ylabel('Error Rate')
plt.title('Errors in Episodes of WCST')
plt.show()

# %% Collect Network Activities

inputs = torch.stack([torch.cat(i, dim=0) for i in inputs], dim=0).transpose(0, 1)
hs = torch.stack([torch.cat(h, dim=0) for h in hs], dim=0).transpose(0, 1) # from batch first to timestep first
# vs = vs.reshape((len_seq//15, (chunks+2)*15, *vs.shape[1:]))
dUs = torch.stack([torch.cat(dU, dim=0) for dU in dUs], dim=0).transpose(0, 1)
# dUs = dUs.reshape((len_seq//15, (chunks+2)*15, *dUs.shape[1:]))
dims = torch.as_tensor(dims).long().transpose(0, 1)
actions = torch.as_tensor(actions).long().transpose(0, 1)
mod_ms = torch.FloatTensor(mod_ms)
mod_ss = torch.FloatTensor(mod_ss)
mod_rs = torch.FloatTensor(mod_rs)
# vis_parafac(hs.reshape((len_seq//15, (chunks+2)*15, *hs.shape[1:])).transpose(0, 1).flatten(1, 2).detach().numpy(), rank=4, plot_type='omni_vec')
# vis_parafac(dUs.reshape((len_seq//15, (chunks+2)*15, *dUs.shape[1:])).transpose(0, 1).flatten(1, 2).detach().numpy(), rank=4, plot_type='omni_mat')
# plt.plot(forbenius_norm(dUs.numpy(), 1))

# %% Plot LDA by subsequence action (Not shown in paper) and by Task

# axe = vis_lda(hs.flatten(0,1), actions.flatten());
# axe.set_title("LDA of Cell State")
# axe = vis_lda(dUs.flatten(2,3).flatten(0,1), actions.flatten());
# axe.set_title("LDA of Fast Weight")

axe = vis_lda(hs.detach().flatten(0, 1), dims.flatten().numpy());
axe.set_title("LDA of Cell State")
axe = vis_lda(dUs.flatten(2,3).detach().flatten(0, 1), dims.flatten().numpy());
axe.set_title("LDA of Fast Weight")

sens = []

# for i in range(num_samples):
#     for j in range(len_seq*(chunks+2)):
#         for k in range(k, len_seq*(chunks+2)):
#             jac = torch.zeros(64, val+bits+1)
#             grad_y = torch.zeros(64)
#             for l in range(64):
#                 grad_y[i] = 1
#                 grad_x = torch.autograd.grad(hs[j][i][l], inputs[i][j], retain_graph=True)
#                 jac.append(grad_x.reshape(val+bits+1))
#                 grad_y[i] = 0



# axe = vis_pca(hs, tags=actions.numpy(), labels=[f"Choice {i}" for i in range(val)]);
# axe.set_title("PCA of Cell State")
# axe = vis_pca(dUs.flatten(2,3), tags=actions.numpy(), labels=[f"Choice {i}" for i in range(val)]);
# axe.set_title("PCA of Fast Weight")

# %% Plot PCA (TCA for fast weight)

axe = vis_pca(hs, tags=dims.numpy(), labels=[f"Task {i}" for i in range(chunks)], threeD=True, data_type='vec');
axe.set_title("PCA of Cell State")
axe = vis_pca(dUs, tags=dims.numpy(), labels=[f"Task {i}" for i in range(chunks)], threeD=True, data_type='mat');
axe.set_title("PCA of Fast Weight")

# %% Decoding Analysis

mean_scores = [];
std_scores = [];

for t in range(75):
    print(t)
    scores_hs_ans = svc_cv(hs[t::75].flatten(0,1), actions[t::75].flatten())
    scores_dUs_ans = svc_cv(dUs.flatten(2,3)[t::75].flatten(0,1), actions[t::75].flatten())
    scores_hs_task = svc_cv(hs[t::75].flatten(0,1), dims[t::75].flatten())
    scores_dUs_task = svc_cv(dUs.flatten(2,3)[t::75].flatten(0,1), dims[t::75].flatten())
    mean_scores.append([scores_hs_ans.mean(), scores_dUs_ans.mean(), scores_hs_task.mean(), scores_dUs_task.mean()])
    std_scores.append([1.96*scores_hs_ans.std(), 1.96*scores_dUs_ans.std(), 1.96*scores_hs_task.std(), 1.96*scores_dUs_task.std()])

mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)

labels = ['v X A', 'dU X A', 'v X T', 'dU X T']
fig, ax = plt.subplots()
for i in range(4):
    eb = plt.errorbar(x=range(75), \
        y=mean_scores[:,i], \
        yerr=std_scores[:,i], label=labels[i]);
plt.ylim(0.0, 1.2)
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
fig.legend(prop={'size': 9})
fig.suptitle('Cross Validation Decoding Accuracy')
fig.show()
