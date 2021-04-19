# %%
from collections import defaultdict
import torch
from modulated_AC import SGRU
import tasks
import torch.optim as optim
import numpy as np
from random import randint
from matplotlib import pyplot as plt
import tqdm
from IPython import display
from fitQ import loglikelihood, fitQ
from scipy import stats
from tabulate import tabulate
from decomposition import *
# %matplotlib qt

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


if torch.cuda.is_available():
    device = torch.device("cuda:0");
else:
    device = torch.device("cpu");

num_ring = 2
ruleset = 'meta'
n_rule = tasks.get_num_rule(ruleset)

n_eachring = 32
n_input, n_output = 1+num_ring*n_eachring, n_eachring+1
hp = {
        # batch size for training
        'batch_size_train': 1,
        # batch_size for testing
        'batch_size_test': 512,
        # input type: normal, multi
        'in_type': 'normal',
        # whether rule and stimulus inputs are represented separately
        'use_separate_input': False,
        # Type of loss functions
        'loss_type': 'lsq',
        # Optimizer
        'optimizer': 'adam',
        # Time constant (ms)
        'tau': 100,
        # discretization time step (ms)
        'dt': 100,
        # discretization time step/time constant
        'alpha': 0.2,
        # input noise
        'sigma_x': 0.0,
        # Stopping performance
        'target_perf': 1.,
        # number of units each ring
        'n_eachring': n_eachring,
        # number of rings
        'num_ring': num_ring,
        # number of rules
        'n_rule': n_rule,
        # first input index for rule units
        'rule_start': 1+num_ring*n_eachring,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of input units
        'ruleset': ruleset,
        # learning rate
        'learning_rate': 0.001,
        }


hp.update({'mix_rule': True, 'l1_h': 0., 'use_separate_input': True})

hp['rng'] = np.random.RandomState(0)

# Rules to train and test. Rules in a set are trained together
    # By default, training all rules available to this ruleset
hp['rule_trains'] = tasks.rules_dict[ruleset]
hp['rules'] = hp['rule_trains']

# Turn into rule_trains format
hp['rule_probs'] = None
# Set default as 1.
rule_prob_map = {'contextdm1': 5, 'contextdm2': 5, 'contextdelaydm1': 5, 'contextdelaydm2': 5,
              'dmsgo': 2, 'dmsnogo': 2, 'dmcgo': 2, 'dmcnogo': 2}
rule_prob = np.array([rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))

model = SGRU(in_type = "continuous",\
             out_type = "binary",\
             num_token = 0,\
             input_dim = n_input+n_output+2,\
             hidden_dim = 128,\
             out_dim = n_output,\
             num_layers = 1,\
             activation="relu",\
             mod_rank = 64,\
             approx_value=False\
            ).to(device);

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=1e-3);

train_epochs = 0000;
val_every = 100;
grad_every = 1; # do gradient accumulation to mediate effect of sampling one task at a time
test_epochs = len(hp['rule_trains'])*4;

cumReward = [];

try:
    state_dict = torch.load("model_multitask", map_location=device);
    model.load_state_dict(state_dict["model_state_dict"]);#print(model.state_dict());
    optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
    cumReward = state_dict["cumReward"];
    print("model loaded successfully")
except:
    print("model failed to load");

print(model);
print(optimizer);

total_loss = 0
total_acc = 0
rule_acc = defaultdict(lambda:[0,0])

for i in tqdm.tqdm(range(train_epochs), position=0, leave=True):
    # initialize loss and reward
    new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=hp['batch_size_train'], device=device);

    rule_train_now = hp['rng'].choice(hp['rule_trains'], p=hp['rule_probs'])
    # Generate a random batch of trials.
    # Each batch has the same trial length
    total_input_support = [];
    for _ in range(5):
        support_trial = tasks.generate_trials(rule_train_now, hp, 'random',
                batch_size=hp['batch_size_train'])

        # support set for quick task adaptation
        inputs_support = torch.from_numpy(support_trial.x)
        targets_support = torch.from_numpy(support_trial.y)

        inputs_support = torch.cat([inputs_support, torch.zeros(1, *inputs_support.shape[1:])], dim=0)
        targets_support = torch.cat([torch.zeros(1, *targets_support.shape[1:]), targets_support], dim=0)

        total_input_support.append(torch.cat([inputs_support, targets_support, \
                                            torch.ones(*inputs_support.shape[:2], 1), \
                                            torch.zeros(*inputs_support.shape[:2], 1)], dim=-1))

    query_trial = tasks.generate_trials(rule_train_now, hp, 'random',
            batch_size=hp['batch_size_train'])

    inputs_query = torch.from_numpy(query_trial.x)
    targets_query = torch.from_numpy(query_trial.y)
    query_mask = torch.from_numpy(query_trial.c_mask)

    total_input_query = torch.cat([inputs_query, torch.zeros_like(targets_query), \
                                        torch.zeros(*inputs_query.shape[:2], 1), \
                                        torch.ones(*inputs_query.shape[:2], 1)], dim=-1)

    total_input = torch.cat([*total_input_support, total_input_query], dim=0)

    new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fws, output, value), mod = model.train().forward(\
                                                          x = total_input.to(device),\
                                                          h = new_h, \
                                                          v = new_v, \
                                                          dU = new_dU, \
                                                          trace = new_trace);

    output = torch.exp(output)[-len(total_input_query):] # log sigmoid to sigmoid
    loss = torch.mean(query_mask.to(device)*((output-targets_query.to(device))**2).flatten(0, 1))

    (loss/grad_every).backward()

    if (i+1)%grad_every==0:
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0);
        optimizer.step();
        optimizer.zero_grad();

    total_loss += loss.item();
    fixation_mask = (torch.argmax(targets_query[-1], dim=-1)==0).float();
        
    angle_diffs = (torch.argmax(output[-1], dim=-1)-torch.argmax(targets_query[-1].to(device), dim=-1))*2*np.pi/n_eachring
    rule_acc[rule_train_now][0] += \
        ((1-fixation_mask)*(tasks.get_dist(angle_diffs.detach().cpu())<=0.2*np.pi).float() +\
          fixation_mask*(output[-1,:,0]>0.5).float().cpu()).mean().item()
    rule_acc[rule_train_now][1] += 1

    # update the policy every [buffer_size] steps
    if (i+1)%val_every==0:
        cumReward.append(dict(rule_acc))
        print(total_loss/val_every)
        print(sorted(rule_acc.items(), key=lambda item: item[1][0]/item[1][1]))
        total_loss = 0;
        total_acc = 0;
        rule_acc = defaultdict(lambda: [0, 0])
        # scheduler1.step();
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'cumReward': cumReward}, 
                    'model_multitask');

hs = [];
dUs = [];
task_type = [];
seq_lens = [];

for i in tqdm.tqdm(range(test_epochs), position=0, leave=True):
    # initialize loss and reward
    new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=hp['batch_size_train'], device=device);

    rule_train_now = hp['rule_trains'][i%len(hp['rule_trains'])]
    # Generate a random batch of trials.
    # Each batch has the same trial length
    total_input_support = [];
    for _ in range(5):
        support_trial = tasks.generate_trials(rule_train_now, hp, 'random',
                batch_size=hp['batch_size_train'])

        # support set for quick task adaptation
        inputs_support = torch.from_numpy(support_trial.x)
        targets_support = torch.from_numpy(support_trial.y)

        inputs_support = torch.cat([inputs_support, torch.zeros(1, *inputs_support.shape[1:])], dim=0)
        targets_support = torch.cat([torch.zeros(1, *targets_support.shape[1:]), targets_support], dim=0)

        total_input_support.append(torch.cat([inputs_support, targets_support, \
                                            torch.ones(*inputs_support.shape[:2], 1), \
                                            torch.zeros(*inputs_support.shape[:2], 1)], dim=-1))

    query_trial = tasks.generate_trials(rule_train_now, hp, 'random',
            batch_size=hp['batch_size_train'])

    inputs_query = torch.from_numpy(query_trial.x)
    targets_query = torch.from_numpy(query_trial.y)
    query_mask = torch.from_numpy(query_trial.c_mask)

    total_input_query = torch.cat([inputs_query, torch.zeros_like(targets_query), \
                                        torch.zeros(*inputs_query.shape[:2], 1), \
                                        torch.ones(*inputs_query.shape[:2], 1)], dim=-1)

    total_input = torch.cat([*total_input_support, total_input_query], dim=0)

    new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fws, output, value), mod = model.train().forward(\
                                                          x = total_input.to(device),\
                                                          h = new_h, \
                                                          v = new_v, \
                                                          dU = new_dU, \
                                                          trace = new_trace);

    output = torch.exp(output)[-len(total_input_query):] # log sigmoid to sigmoid
    
    hs.append(last_layer_out)
    dUs.append(last_layer_fws)
    seq_lens.append(total_input.shape[0])
    task_type.append(i%len(hp['rule_trains']))

    fixation_mask = (torch.argmax(targets_query[-1], dim=-1)==0).float();
        
    angle_diffs = (torch.argmax(output[-1], dim=-1)-torch.argmax(targets_query[-1].to(device), dim=-1))*2*np.pi/n_eachring
    rule_acc[rule_train_now][0] += \
        ((1-fixation_mask)*(tasks.get_dist(angle_diffs.detach().cpu())<=0.2*np.pi).float() +\
          fixation_mask*(output[-1,:,0]>0.5).float().cpu()).mean().item()
    rule_acc[rule_train_now][1] += 1



hs = torch.cat(hs).squeeze()
dUs = torch.cat(dUs).squeeze()

tags = [task_type[i] for i in range(test_epochs) for _ in range(seq_lens[i])]

vis_pca(hs, tags, tags, data_type='vec')
vis_pca(dUs, tags, tags, data_type='vec')