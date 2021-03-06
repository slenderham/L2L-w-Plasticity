# %%
import torch
from modulated_AC import SGRU
from TMaze import TMaze, Trajectories
import torch.optim as optim
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from ppo import PPO, Memory
import tqdm
from IPython import display
from fitQ import loglikelihood, fitQ
from scipy import stats
from tabulate import tabulate
from decomposition import forbenius_norm, vis_lda, vis_pca
%matplotlib qt

torch.manual_seed(0);
np.random.seed(0);

def add_weight_decay(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if ("ln" in name or "encoder" in name or "weight" not in name or "mod" in name):
            no_decay.append(param);
        else:
            decay.append(param);

    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 0.}];

action_size = 4;
obs_window_size = 9;

if torch.cuda.is_available():
    device = torch.device("cuda:0");
else:
    device = torch.device("cpu");

model = SGRU(in_type = "continuous",\
             out_type = "categorical",\
             num_token = 0,\
             input_dim = obs_window_size+action_size+1,\
             hidden_dim = 64,\
             out_dim = action_size,\
             num_layers = 1,\
             activation="relu",\
             mod_rank = 16\
            ).to(device);

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=1e-3);
# scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5);

train_epochs = 0;
val_epochs = 30;
buffer_size = 50;
max_steps = 600;

episode_buffer = Memory();
cumReward = [];

maze = TMaze(height = 4, 
             width = 2,
             max_steps = max_steps,
             switch_after = 15);

# try:
state_dict = torch.load("model_maze", map_location=device);
model.load_state_dict(state_dict["model_state_dict"]);#print(model.state_dict());
optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
cumReward = state_dict["cumReward"];
# model.rnns[0].alpha = torch.nn.Parameter(-torch.ones(1)*1e6)
print("model loaded successfully")
# except:
    # print("model failed to load");

ppo = PPO(policy = model, \
          optimizer = optimizer, \
          seq_len = max_steps,\
          buffer_size = buffer_size,\
          beta_v = 0.4,\
          beta_entropy = max(0, 1-0.01*len(cumReward)),\
          gamma = 0.95,\
          lambd = 0.95
         );

print(model);
print(optimizer);

for i in tqdm.tqdm(range(train_epochs), position=0, leave=True):
    # initialize loss and reward
    maze.reset();
    reward = torch.zeros(1, 1, device=device);
    new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
    action = torch.zeros(1, action_size, device=device);
    obs = maze._get_obs();
    done = False;

    episode_buffer.actions.append([]);
    episode_buffer.states.append([]);
    episode_buffer.logprobs.append([]);
    episode_buffer.rewards.append([]);
    episode_buffer.values.append([]);

    with torch.no_grad():
        while(not done):
            total_input = torch.cat(
                                    (torch.as_tensor(obs, dtype=torch.float).flatten().unsqueeze(0).to(device), action.to(device), torch.as_tensor(reward).reshape(1,1).to(device)), dim=1
                                ).unsqueeze(0);
            # one iter of network, notice that the reward is from the previous time step
            new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fws, log_probs, value), mod = model.train().forward(\
                                                          x = total_input.to(device),\
                                                          h = new_h, \
                                                          v = new_v, \
                                                          dU = new_dU, \
                                                          trace = new_trace);

            # sample an action
            m = torch.distributions.Categorical(logits = log_probs[-1]);
            action_idx = m.sample();
            action = torch.zeros(1, action_size, dtype=torch.float, device=device);
            action[:, action_idx] = 1.0;

            obs, reward, done, info = maze.step(action_idx.item());

            episode_buffer.actions[-1].append(action_idx);
            episode_buffer.states[-1].append(total_input); # batch_size(1), trial number, within trial time step, 1, input_dim
            episode_buffer.logprobs[-1].append(m.log_prob(action_idx));
            episode_buffer.rewards[-1].append(reward);
            episode_buffer.values[-1].append(value[-1]);

            # maze.render();
            # display.clear_output(wait=True);
            # display.display(plt.gcf())

    # update the policy every [buffer_size] steps
    if (i+1)%buffer_size==0:
        ppo.update(episode_buffer);
        cumReward.append(torch.sum(torch.as_tensor(episode_buffer.rewards)));
        print(cumReward[-1]);
        ppo.beta_entropy = max(ppo.beta_entropy-0.01, 0.);
        episode_buffer.clear_memory();
        # scheduler1.step();
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'cumReward': cumReward}, 
                    'model_maze');


def evaluate():

    states = [];
    trajectory = Trajectories(maze);
    # time, h, mod, choice point or not, which goal, origin or not, 
    
    for i in tqdm.tqdm(range(val_epochs), position=0, leave=True):
        # initialize loss and reward
        maze.reset();
        reward = torch.zeros(1, 1, device=device);
        new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=1, device=device);
        action = torch.zeros(1, action_size, device=device);
        obs = maze._get_obs();
        done = False;
        trajectory.add_new_episode();
        states.append([[new_v[0], new_dU[0], torch.zeros(1, 16), torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)]]);

        with torch.no_grad():
            while(not done):
                total_input = torch.cat(
                                        (torch.as_tensor(obs, dtype=torch.float).flatten().unsqueeze(0).to(device), action.to(device), torch.as_tensor(reward).reshape(1,1).to(device)), dim=1
                                    ).unsqueeze(0);
                # one iter of network, notice that the reward is from the previous time step
                new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fws, log_probs, value), (mod, mod_e, mod_m, mod_r) = model.eval().forward(\
                                                            x = total_input.to(device),\
                                                            h = new_h, \
                                                            v = new_v, \
                                                            dU = new_dU, \
                                                            trace = new_trace);

                states[-1].append([new_v[0], new_dU[0], mod[0], mod_e[0], mod_m[0], mod_r[0]]);

                # sample an action
                m = torch.distributions.Categorical(logits = log_probs[-1]);
                action_idx = m.sample();
                action = torch.zeros(1, action_size, dtype=torch.float, device=device);
                action[:, action_idx] = 1.0;

                obs, reward, done, info = maze.step(action_idx.item());
                trajectory.add_info(info);

    actions, rewards = trajectory.get_choices_and_outcome();
    alphabetaeps, Qls, Qrs, nll = fitQ(actions, rewards);

    # trajectory.get_feats(Qls, Qrs)

    return alphabetaeps, Qls, Qrs, nll, states, actions, rewards, trajectory

abe, Qls, Qrs, nll, states, actions, rewards, trajectory = evaluate();
print(abe, nll)

mods = [[j[2].squeeze().detach().numpy() for j in t] for t in states]
vs = [[j[0].squeeze().detach().numpy() for j in t] for t in states]
dUs = [[j[1].squeeze().detach().numpy() for j in t] for t in states]
mod_ss = [[j[3].squeeze().detach().numpy().reshape(1) for j in t] for t in states]
mod_ms = [[j[4].squeeze().detach().numpy().reshape(1) for j in t] for t in states]
mod_rs = [[j[5].squeeze().detach().numpy().reshape(1) for j in t] for t in states]

feats = trajectory.get_feats(Qls, Qrs, abe);
results, dictvec, feats_flat, acts_flat = trajectory.linear_regression_fit(feats, mod_ms);

# axe = vis_pca(torch.tensor(vs).flatten(0,1), 2*np.array(trajectory.get_task()).flatten()+np.array(trajectory.get_stage()).flatten(), labels=['Left+Approach','Left+Return','Right+Approach','Right+Return']);
# axe.set_title("PCA of Cell State")
# plt.show()
# axe = vis_pca(torch.tensor(dUs).flatten(2,3).flatten(0,1), 2*np.array(trajectory.get_task()).flatten()+np.array(trajectory.get_stage()).flatten(), labels=['Left+Approach','Left+Return','Right+Approach','Right+Return']);
# axe.set_title("PCA of Fast Weight")
# plt.show()

headers = ['Location', "intercept"]
headers.extend(dictvec.feature_names_);
results_to_print = [];
# vars_to_plot = ["prev_outcome", "q_prev_c", "prev_choice", "upcoming_choice"]
# var_names = dict(zip(vars_to_plot, ['Previous Outcome', 'Q of Previous Choice', 'Previous Choice', 'Upcoming Choice']))
vars_to_plot = ["rpe", "updated_qc", "prev_choice", "upcoming_choice"]
var_names = dict(zip(vars_to_plot, ['Reward Prediction Error', 'Updated Q', 'Previous Choice', 'Upcoming Choice']))
fig, axes = plt.subplots(2, 2)
i = 0;
loc_names = {
    (2, 3): 'S',
    (1, 3): 'G1',
    (1, 2): 'R',
    (4, 3): 'G2'
}
locs = [(2, 3), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (4, 3), (3, 3)]
tick_labels = [loc_names.get(k, ' ') for k in locs];


for v in vars_to_plot:
    values = []
    cis = []
    for k in locs:
        results_to_print.append([k]);
        results_to_print[-1].extend(['{:.2f} ± {:.2f} (p={:.3f})'.format(m, 1.96*s, p) for (m, s, p) in zip(results[k][0].params, results[k][0].bse, results[k][0].pvalues)])
        values.append(results[k][0].params[v])
        cis.append(results[k][0].bse[v])
    values = np.array(values)
    cis = np.array(cis)
    axes[i//2][i%2].bar(list(range(len(values))), values, color=plt.get_cmap('tab10').colors[i], alpha=0.5)
    axes[i//2][i%2].errorbar(list(range(len(values))), values, 1.96*cis, capsize=1, c=plt.get_cmap('tab10').colors[i], fmt='.')
    axes[i//2][i%2].set_xticks(list(range(len(values))))
    axes[i//2][i%2].set_xticklabels(tick_labels)
    axes[i//2][i%2].set_title(var_names[v])
    i += 1;

# print(tabulate(results_to_print, headers=headers, tablefmt="github"))
fig.show()


# for k in results.keys():
#     print(k)
#     print(results[k][0].summary())

# h2modweight = model.rnns[0].mod2h.weight[1].flatten().detach().numpy();

# fig, axes = plt.subplots(2, 10)

# pos = sorted(results.keys());

# for j in range(10):
#     sig_r_coeff_pos = [i for i in range(16) if results[pos[j]][i].pvalues[1]<0.05];
#     sig_q_coeff_pos = [i for i in range(16) if results[pos[j]][i].pvalues[3]<0.05];
#     r_coeffs = [results[pos[j]][i].params[1]  for i in sig_r_coeff_pos];
#     q_coeffs = [results[pos[j]][i].params[3] for i in sig_q_coeff_pos];
#     print(pos[j])
#     try:
#         axes[0, j].scatter(h2modweight[sig_r_coeff_pos], r_coeffs);
#         axes[0, j].plot(h2modweight[sig_r_coeff_pos], np.poly1d(np.polyfit(h2modweight[sig_r_coeff_pos], r_coeffs, 1))(h2modweight[sig_r_coeff_pos]));
#         print(stats.pearsonr(h2modweight[sig_r_coeff_pos], r_coeffs));
#         axes[1, j].scatter(h2modweight[sig_q_coeff_pos], q_coeffs);
#         axes[1, j].plot(h2modweight[sig_q_coeff_pos], np.poly1d(np.polyfit(h2modweight[sig_q_coeff_pos], q_coeffs, 1))(h2modweight[sig_q_coeff_pos]));
#         print(stats.pearsonr(h2modweight[sig_q_coeff_pos], q_coeffs));
#     except:
#         None


# %%
