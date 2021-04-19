# %%
import torch
from torchvision import datasets, transforms
from modulated_AC import SGRU
import torch.optim as optim
import numpy as np
from random import randint
from ppo import PPO, Memory
from matplotlib import pyplot as plt
from lamb import Lamb, get_cosine_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
from lamb import Lamb
import pickle
# from decomposition import *
from fitQ import fitCausal
from scipy.stats import spearmanr, pearsonr
%matplotlib qt

from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine
from torchmeta.utils.data import BatchMetaDataLoader

seed=randint(0, 100);
# seed=83;
print(seed);
torch.manual_seed(seed);
np.random.seed(seed);

def add_weight_decay(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if ("ln" in name or "encoder" in name or "mod" in name or "bn" in name or "weight" not in name):
            no_decay.append(param);
        else:
            decay.append(param);

    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 0.}];

def offset(batch):
    '''
    (1) [num_pics_per_trial] images
    (2) outcome
    (3) repeat (1)(2) for [num_trials]
    (4) present each image that was used
    (5) calculate reward

    outputs:
        images: ((num_pics_per_trial+1)*num_trials) X batch_size X {image_dims...}, each trial consists of the images and a 
        outcomes_total: ((num_pics_per_trial+1)*num_trials) X batch_size X 1
        is_blank_image: ((num_pics_per_trial+1)*num_trials) X batch_size X 1
    '''

    global batch_size, num_pics, num_trials, num_pics_per_trial, num_repeats, freqs, img_size, device;

    task_types = torch.round(torch.rand(batch_size)).long(); # get task type, either novel image to novel outcome, or novel image to non-novel outcome
    novel_outcome = torch.round(torch.rand(batch_size)).float()*2.0-1.0; # get the novel outcome (good or bad)
    # novel_outcome = torch.ones(batch_size)

    # batch X num_pics X 1 X img_size X img_size
    train_inputs_orig, _ = batch["train"]
    train_inputs_orig = (train_inputs_orig-train_inputs_orig.mean())/(train_inputs_orig.std()+1e-8);
    # get the image input: [num_pics] different omniglot pictures, 
    perm_order = torch.argsort(torch.rand(batch_size, num_pics_per_trial*num_trials), dim=-1).to(device);
    for i in range(batch_size):
        if perm_order[i, -1]//num_pics_per_trial < num_trials-2:
            for j in range(num_pics_per_trial*num_trials):
                if perm_order[i, j]//num_pics_per_trial >= num_trials-2:
                    perm_order[[i,i], [j,-1]] = perm_order[[i,i], [-1,j]]
                    break;

    orders = torch.zeros(batch_size, num_pics_per_trial*num_trials);
    cum_freq = 0;
    for i, f in enumerate(freqs):
        for j in range(batch_size):
            orders[[j]*f, perm_order[j, cum_freq:cum_freq+f]] = i;
        cum_freq += f;
    assert(cum_freq==sum(freqs));

    bonus_round_idx = torch.argsort(torch.rand(size=(batch_size, num_pics)), dim=-1);
    bonus_round_novel_outcome_idx = torch.argmax(bonus_round_idx, dim=-1);

    # shuffle input
    def shuffle_input(train_inputs):
        train_inputs_shuff = torch.zeros(batch_size, num_pics_per_trial*num_trials, 64).to(train_inputs.device);
        for i in range(batch_size):
            train_inputs_shuff[i] = train_inputs[[i]*num_pics_per_trial*num_trials, orders[i].long().to(train_inputs.device)];
        assert(train_inputs_shuff.shape==(batch_size, num_pics_per_trial*num_trials, 64))
        # separate the sequence into trials, and add blank screen after each trial
        train_inputs_shuff = train_inputs_shuff.reshape(batch_size, num_trials, num_pics_per_trial, 64);
        train_inputs_shuff = torch.cat([train_inputs_shuff, torch.zeros(batch_size, num_trials, 1, 64).to(train_inputs.device)], dim=2)
        train_inputs_shuff = train_inputs_shuff.flatten(1, 2)
        
        # add each presented image after entire trial to query for rating (each for a few timesteps)
        bonus_round_train_inputs = torch.zeros_like(train_inputs);
        for i in range(batch_size):
            bonus_round_train_inputs[i] = train_inputs[[i]*num_pics, bonus_round_idx[i].to(train_inputs.device)];
        
        train_inputs_shuff = torch.cat([train_inputs_shuff, bonus_round_train_inputs, torch.zeros(batch_size, 1, 64).to(train_inputs.device)], dim=1); 
        train_inputs_shuff = train_inputs_shuff.transpose(0, 1);
        train_inputs_shuff = torch.repeat_interleave(train_inputs_shuff, repeats=num_repeats, dim=0)
        return train_inputs_shuff

    # fig, axes = plt.subplots(num_trials, num_pics_per_trial+1)
    # for i in range(num_trials):
    #     for j in range(num_pics_per_trial+1):
    #         axes[i][j].imshow(train_inputs_shuff[i*(num_pics_per_trial+1)+j,0,0])
    # plt.show()
    # fig, axes = plt.subplots(num_pics)
    # for i in range(num_pics):
    #     axes[i].imshow(train_inputs_shuff[num_trials*(num_pics_per_trial+1)+i,0,0])
    # plt.show()
    # num_repeats* (num_trials*(num_pics_per_trial+1)+num_pics+1), batch_size, 1, img_size, img_size
    # train_inputs_shuff = (train_inputs_shuff-train_inputs_shuff.mean())/(train_inputs_shuff.std()+1e-8);

    # calculate outcomes based on task type
    outcomes = torch.zeros(batch_size, num_trials, 1);
    novel_stim_loc = perm_order[:,-1]//num_pics_per_trial;
    for i in range(batch_size):
        # if novel image to novel outcome, find novel outcome
        if task_types[i]==0:
            outcomes[i] = -novel_outcome[i]; # set all others to be negative of novel outcome
            outcomes[i, novel_stim_loc[i], 0] = novel_outcome[i]; # if novel stim-novel outcome and the trial contains the novel stim, 
        # if novel image to nonnovel outcome, sample a trial to put novel outcome
        else:
            outcomes[i] = -novel_outcome[i]; # set all others to be negative of novel outcome
            # novel_outcome_loc = [t for t in range(num_trials) if t!=novel_stim_loc[i]][torch.randperm(num_trials-1)[0]]
            novel_outcome_loc = novel_stim_loc[i]-1 if novel_stim_loc[i]==num_trials-1 else novel_stim_loc[i]+1
            assert(novel_outcome_loc in (num_trials-1, num_trials-2))
            outcomes[i, novel_outcome_loc, 0] = novel_outcome[i];
    
    outcomes_total = torch.zeros(batch_size, num_trials, num_pics_per_trial, 1) 
    outcomes_total = torch.cat([outcomes_total, outcomes.reshape(batch_size, num_trials, 1, 1)], dim=2);
    outcomes_total = outcomes_total.flatten(1, 2); # num_trials * (num_pics_per_trial + 1)
    outcomes_total = torch.cat([outcomes_total, torch.zeros(batch_size, (num_pics+1), 1)], dim=1); # num_trials * (num_pics_per_trial + 1) + num_pics + 1
    outcomes_total = outcomes_total.transpose(0, 1);

    outcomes_total = torch.repeat_interleave(outcomes_total, repeats=num_repeats, dim=0)

    # indicator for phase of task (image, outcome after each trial, bonus round, value estimation)
    phase_ind = torch.zeros(batch_size, num_trials*(num_pics_per_trial+1)+num_pics+1, 4);
    for j in range(num_trials):
        phase_ind[:, j*(num_pics_per_trial+1):j*(num_pics_per_trial+1)+num_pics_per_trial, 0] = 1.0;
        phase_ind[:, j*(num_pics_per_trial+1)+num_pics_per_trial, 1] = 1.0
    phase_ind[:, num_trials*(num_pics_per_trial+1):num_trials*(num_pics_per_trial+1)+num_pics, 2] = 1.0
    phase_ind[:, num_trials*(num_pics_per_trial+1)+num_pics:, 3] = 1.0
    phase_ind = phase_ind.transpose(0, 1)
    phase_ind = torch.repeat_interleave(phase_ind, repeats=num_repeats, dim=0)

    # print(outcomes_total[:,0].flatten())
    # print(task_types[0])
    # print(novel_outcome[0])
    # print(bonus_round_novel_outcome_idx[0])

    assert(outcomes_total.shape[:2]==phase_ind.shape[:2])

    return [train_inputs_orig.to(device), torch.cat([phase_ind, outcomes_total], dim=-1).to(device), shuffle_input], \
            orders.to(device), outcomes.to(device), novel_stim_loc.to(device),\
            bonus_round_idx.to(device), bonus_round_novel_outcome_idx.to(device), \
            task_types.to(device), novel_outcome.to(device);

batch_size = 2;
num_pics = 3;
len_seq = 1;
num_trials = 5;
num_pics_per_trial = 5;
num_repeats = 3;
freqs = [16, 8, 1];
assert(sum(freqs)==num_trials*num_pics_per_trial)
assert(len(freqs)==num_pics)
assert(all([freqs[i]>=freqs[i+1] for i in range(num_pics-1)]));
img_size = 28;
train_batches = 0000;
val_batches = 25;
val_every = 25;
test_batches = 10;
assert(val_every%len_seq==0)
task_type_name = ["Novel Image -> Novel Outcome", "Novel Image -> Nonnovel Outcome"]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu");

# RandomAffine(degrees=11.25, translate=(0.1, 0.1)),

train_data = Omniglot("data",
                         num_classes_per_task=num_pics,
                         transform=Compose([Resize(img_size, interpolation=Image.LANCZOS),  ToTensor()]),
                         target_transform=Categorical(num_classes=num_pics),
                         class_augmentations=[Rotation([90, 180, 270])],
                         meta_train=True,
                         download=True);
train_data = ClassSplitter(train_data, shuffle=True, num_train_per_class=1, num_test_per_class=1);

val_data = Omniglot("data",
                         num_classes_per_task=num_pics,
                         transform=Compose([Resize(img_size, interpolation=Image.LANCZOS), ToTensor()]),
                         target_transform=Categorical(num_classes=num_pics),
                         meta_val=True,
                         download=True);
val_data = ClassSplitter(val_data, shuffle=True, num_train_per_class=1, num_test_per_class=1);

test_data = Omniglot("data",
                         num_classes_per_task=num_pics,
                         transform=Compose([Resize(img_size, interpolation=Image.LANCZOS), ToTensor()]),
                         target_transform=Categorical(num_classes=num_pics),
                         meta_test=True,
                         download=True);
test_data = ClassSplitter(test_data, shuffle=True, num_train_per_class=1, num_test_per_class=1);

lr = 1e-3;
clip = 1.0;
clip_val = 1.0;

train_iter = BatchMetaDataLoader(train_data, batch_size=batch_size);
val_iter = BatchMetaDataLoader(val_data, batch_size=batch_size);
test_iter = BatchMetaDataLoader(test_data, batch_size=batch_size);

model = SGRU(in_type = "image+continuous",\
            out_type = "continuous",\
            num_token = 0,\
            input_dim = 5,\
            hidden_dim = 256,\
            out_dim = 1,\
            num_layers = 1,\
            activation="relu",\
            mod_rank = 64).to(device);

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=lr, eps=1e-4);
scheduler1 = optim.lr_scheduler.StepLR(optimizer, 6000, 0.1)
cumReward = []
try:
    state_dict = torch.load("model_one_shot-99", map_location=device);
    print(model.load_state_dict(state_dict["model_state_dict"]));
    optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
    scheduler1.load_state_dict(state_dict["scheduler_state_dict"]);
    cumReward = state_dict['cumReward'];print(cumReward[-1])
    print("model loaded successfully");
except:
    print("model failed to load");

print(model);
print(optimizer);
print(scheduler1.state_dict().keys());

episode_buffer = Memory()
ppo = PPO(policy = model, \
          optimizer = optimizer, \
          seq_len = 1,\
          buffer_size = batch_size,\
          beta_v = 0.4,\
          beta_entropy = 0.001,\
          gamma = 0.95,\
          lambd = 0.95
         );

train_iter = enumerate(train_iter);
val_iter = enumerate(val_iter);
test_iter = enumerate(test_iter);

for idx, batch in tqdm(train_iter, position=0):
    if train_batches==0:
        break;
    with torch.no_grad():
        input_total, orders, outcomes, novel_stim_loc, bonus_round_orders, bonus_round_novel_outcome_idx, task_types, novel_outcome = offset(batch);
        new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
        new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fw, output, value), mod = model.train().forward(\
                                                                                x = input_total,\
                                                                                h = new_h, \
                                                                                v = new_v, \
                                                                                dU = new_dU, \
                                                                                trace = new_trace);

        # sample an action
        log_probs = output[-(num_pics)*num_repeats-1:-num_repeats:num_repeats]; # should be num_pics X batch_size X 1
        log_probs = log_probs.squeeze(-1).t();
        log_probs = torch.nn.functional.log_softmax(log_probs, -1);

        m = torch.distributions.Categorical(logits = log_probs);
        action_idx = m.sample();

        # get reward
        got_novel_trial = (torch.rand(batch_size)>0.4).to(device);
        # got_novel_trial = (torch.ones(batch_size)>0.5).to(device)
        chose_novel = got_novel_trial & (action_idx==bonus_round_novel_outcome_idx)
        # novel stim to novel outcome -> choose novel then novel outcome, didn't choose novel then nonnovel outcome
        # novel stim to nonnovel outcome -> choose novel then nonnovel outcome, didn't choose novel then novel outcome
        reward = (1-task_types)*(chose_novel).float()*novel_outcome\
                +(1-task_types)*(~chose_novel).float()*(-novel_outcome)\
                +(task_types)*(chose_novel).float()*(-novel_outcome)\
                +(task_types)*(~chose_novel).float()*(novel_outcome);

        episode_buffer.actions.append(action_idx); # time_step, batch size, 1
        episode_buffer.states.append(input_total); # trial number, within trial time step, batch_size, 1, input_dim
        episode_buffer.logprobs.append(m.log_prob(action_idx));
        episode_buffer.rewards.append(reward);
        episode_buffer.values.append(value[-1]);

    ppo.update(episode_buffer, task='one_shot', num_pics=num_pics, num_repeats=num_repeats);
    episode_buffer.clear_memory();
    scheduler1.step();

    if (idx+1)%val_every==0:
        with torch.no_grad():
            valReward = 0;
            for jdx, batch in tqdm(val_iter, position=0):
                input_total, orders, outcomes, novel_stim_loc, bonus_round_orders, bonus_round_novel_outcome_idx, task_types, novel_outcome = offset(batch);
                new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
                new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fw, output, value), mod = model.eval().forward(\
                                                                                        x = input_total,\
                                                                                        h = new_h, \
                                                                                        v = new_v, \
                                                                                        dU = new_dU, \
                                                                                        trace = new_trace);

                # sample an action
                log_probs = output[-num_pics*num_repeats-1:-num_repeats:num_repeats]; # should be num_pics X batch_size X 1
                log_probs = log_probs.squeeze(-1).t();
                log_probs = torch.nn.functional.log_softmax(log_probs, -1);
                m = torch.distributions.Categorical(logits = log_probs);
                action_idx = m.sample();

                # get reward
                got_novel_trial = (torch.rand(batch_size)>0.4).to(device);
                # got_novel_trial = (torch.ones(batch_size)>0.5).to(device)
                chose_novel = got_novel_trial & (action_idx==bonus_round_novel_outcome_idx)
                reward = (1-task_types)*(chose_novel).float()*novel_outcome\
                        +(1-task_types)*(~chose_novel).float()*(-novel_outcome)\
                        +(task_types)*(chose_novel).float()*(-novel_outcome)\
                        +(task_types)*(~chose_novel).float()*(novel_outcome);
                valReward += reward.mean()/val_batches;
                if ((jdx+1)%val_batches==0):
                    print(valReward)
                    cumReward.append(valReward);
                    torch.save({'model_state_dict': model.state_dict(), \
                                'optimizer_state_dict': optimizer.state_dict(), \
                                'scheduler_state_dict': scheduler1.state_dict(), \
                                'cumReward': cumReward}, 'model_one_shot');
                    break;

    if (idx+1)%train_batches==0:
        print('training complete, proceeding to test')
        break;

all_task_types = [];
all_novel_outcomes = [];
all_actions = [];
all_bonus_round_stim_orders = [];
all_bonus_round_novel_outcome_idx = [];
all_stim_orders = []
all_outcomes = []
all_novel_stim_loc = []
hs = [];
dUs = [];
ms = [];
ss = [];
rs = [];
os = [];
ratings = [];
testCorrects = [];

with torch.no_grad():
    testReward = 0;
    for jdx, batch in tqdm(test_iter, position=0):
        input_total, orders, outcomes, novel_stim_loc, bonus_round_orders, bonus_round_novel_outcome_idx, task_types, novel_outcome = offset(batch);
        new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
        new_v, new_h, new_dU, new_trace, (last_layer_out, last_layer_fw, output, value), mod = model.eval().forward(\
                                                                                x = input_total,\
                                                                                h = new_h, \
                                                                                v = new_v, \
                                                                                dU = new_dU, \
                                                                                trace = new_trace);

        # sample an action
        log_probs = output[-num_pics*num_repeats-1:-num_repeats:num_repeats]; # should be num_pics X batch_size X 1
        log_probs = log_probs.squeeze(-1).t();
        ratings.append(log_probs);
        log_probs = torch.nn.functional.log_softmax(log_probs, -1);
        m = torch.distributions.Categorical(logits = log_probs);
        action_idx = m.sample();

        # get reward
        got_novel_trial = (torch.rand(batch_size)>0.4).to(device);
        chose_novel = got_novel_trial & (action_idx==bonus_round_novel_outcome_idx)
        reward = (1-task_types)*(chose_novel).float()*novel_outcome\
                +(1-task_types)*(~chose_novel).float()*(-novel_outcome)\
                +(task_types)*(chose_novel).float()*(-novel_outcome)\
                +(task_types)*(~chose_novel).float()*(novel_outcome);
        get_rwd = (1-task_types)*(action_idx==bonus_round_novel_outcome_idx).float()*novel_outcome\
                +(1-task_types)*(action_idx!=bonus_round_novel_outcome_idx).float()*(-novel_outcome)\
                +(task_types)*(action_idx==bonus_round_novel_outcome_idx).float()*(-novel_outcome)\
                +(task_types)*(action_idx!=bonus_round_novel_outcome_idx).float()*(novel_outcome);
        testReward += reward.mean()/test_batches;
        # testCorrect += get_rwd.mean()/test_batches;
        testCorrects += get_rwd.flatten().tolist()

        all_task_types.append(task_types);
        all_novel_outcomes.append(novel_outcome);
        all_actions.append(action_idx);
        all_bonus_round_novel_outcome_idx.append(bonus_round_novel_outcome_idx);
        all_bonus_round_stim_orders.append(bonus_round_orders);
        all_stim_orders.append(orders.reshape(batch_size, num_trials, num_pics_per_trial))
        all_novel_stim_loc.append(novel_stim_loc);
        all_outcomes.append(outcomes.squeeze(-1))
        hs.append(last_layer_out)
        dUs.append(last_layer_fw)
        ms.append(mod[2])
        ss.append(mod[1])
        rs.append(mod[3])
        os.append(mod[4])

        if (jdx+1)%test_batches==0:
            print(testReward)
            testCorrects = (np.array(testCorrects)+1)/2
            print(np.mean(testCorrects))
            print(np.std(testCorrects)*1.96/np.array(test_batches))
            print('testing complete')
            break;

all_task_types = torch.cat(all_task_types, dim=0)
all_novel_outcomes = torch.cat(all_novel_outcomes, dim=0)
all_actions = torch.cat(all_actions, dim=0)
all_stim_orders = torch.cat(all_stim_orders, dim=0)
ratings = (torch.cat(ratings, dim=0)/10).softmax(-1) # normalize to rating between 0 and 1, soften with higher temperature
causal_ratings = (all_novel_outcomes<0).float().unsqueeze(-1)*ratings \
               + (all_novel_outcomes>0).float().unsqueeze(-1)*(1-ratings)/2
all_bonus_round_novel_outcome_idx = torch.cat(all_bonus_round_novel_outcome_idx, dim=0) 
all_outcomes = torch.cat(all_outcomes, dim=0)
all_bonus_round_stim_orders = torch.cat(all_bonus_round_stim_orders, dim=0)
all_novel_stim_loc = torch.cat(all_novel_stim_loc, dim=0)
hs = torch.cat(hs, dim=1)
dUs = torch.cat(dUs, dim=1)
ms = torch.cat(ms, dim=1)
ss = torch.cat(ss, dim=1)
rs = torch.cat(rs, dim=1)
os = torch.cat(os, dim=1)

# deltasgammatau, lrs, all_alphas, mse = fitCausal(all_stim_orders, all_outcomes, ratings, all_bonus_round_stim_orders, prior=[1, 1, 1]);
# deltasgammatau, lrs, all_alphas, mse = fitCausal(all_stim_orders, all_outcomes==all_novel_outcomes.unsqueeze(-1), causal_ratings, all_bonus_round_stim_orders, prior=[1, 1, 1]);
deltasgammatau, lrs, all_alphas, mse = fitCausal(all_stim_orders, -2*(all_outcomes==all_novel_outcomes.unsqueeze(-1))+1, causal_ratings, all_bonus_round_stim_orders);

sum_alphas = all_alphas.sum(-1, keepdims=True)
all_means = all_alphas/sum_alphas
all_vars = all_alphas*(sum_alphas-all_alphas)/(sum_alphas**2*(sum_alphas+1))
causal_unc = all_vars.sum(-1)

actual_lrs = torch.split(ms.squeeze(), [(num_pics_per_trial+1)*num_repeats]*num_trials+[num_pics*num_repeats]+[num_repeats], dim=0)
actual_lrs = model.rnns[0].tau_U.sigmoid().detach()*torch.stack(actual_lrs[:num_trials])[:,-num_repeats:].mean(1); # -> num_trials X batch size
# actual_lrs = (actual_lrs-actual_lrs.mean())/(actual_lrs.std()+1e-6)
fig, axes = plt.subplots(1, 3)
delta_unc = causal_unc[:, 1:]-causal_unc[:, :-1]
axes[0].plot(np.unique(causal_unc[:, :-1].flatten()), np.poly1d(np.polyfit(causal_unc[:, :-1].flatten(), actual_lrs.t().flatten(), 1))(np.unique(causal_unc[:, :-1])), c='black')
axes[0].scatter(causal_unc[:, :-1].flatten(), actual_lrs.t().flatten(), c='lightskyblue', alpha=0.5)
r_val, p_val = spearmanr(causal_unc[:, :-1].flatten(), actual_lrs.t().flatten())
# axes[0].set_ylim([-1, 10])
# axes[0].text(0.22, 0, f'r={r_val:.3f}, p={p_val:.3f}')
axes[0].set_xlabel('Causal Uncertainty Before Trial')
axes[0].set_ylabel('Modulation Signal')
axes[0].set_aspect('auto')

axes[1].plot(np.unique(causal_unc[:, 1:].flatten()), np.poly1d(np.polyfit(causal_unc[:, 1:].flatten(), actual_lrs.t().flatten(), 1))(np.unique(causal_unc[:, 1:])), c='black')
axes[1].scatter(causal_unc[:, 1:].flatten(), actual_lrs.t().flatten(), c='lightgreen', alpha=0.5)
r_val, p_val = spearmanr(causal_unc[:, 1:].flatten(), actual_lrs.t().flatten())
# axes[1].text(0.22, 0, f'r={r_val:.3f}, p={p_val:.3f}')
# axes[1].set_ylim([-1, 10])
axes[1].set_xlabel('Causal Uncertainty After Trial')
axes[1].set_ylabel('Modulation Signal')
axes[1].set_aspect('auto')

axes[2].plot(np.unique(delta_unc.flatten()), np.poly1d(np.polyfit(delta_unc.flatten(), actual_lrs.t().flatten(), 1))(np.unique(delta_unc)), c='black')
axes[2].scatter(delta_unc.flatten(), actual_lrs.t().flatten(), c='lightcoral', alpha=0.5)
r_val, p_val = spearmanr(delta_unc.flatten(), actual_lrs.t().flatten())
# axes[2].text(0.0, 0, f'r={r_val:.3f}, p={p_val:.3f}')
# axes[2].set_ylim([-1, 10])
axes[2].set_xlabel('Changes in Causal Uncertainty')
axes[2].set_ylabel('Modulation Signal')
axes[2].set_aspect('auto')

dws = (dUs[1:]-dUs[:-1]).pow(2).mean([2,3])
# plt.plot(dws)
# plt.xlabel('Timestep')
# plt.ylabel(r'$\Delta dU$')
actual_dws = torch.split(dws.squeeze(), [(num_pics_per_trial+1)*num_repeats]*num_trials+[num_pics*num_repeats]+[num_repeats-1], dim=0)
actual_dws = torch.stack(actual_dws[:num_trials])[:,-num_repeats:].mean(1); # -> num_trials X batch size
actual_dws = actual_dws.log()

fig, axes = plt.subplots(1, 3)
delta_unc = causal_unc[:, 1:]-causal_unc[:, :-1]
axes[0].plot(np.unique(causal_unc[:, :-1].flatten()), np.poly1d(np.polyfit(causal_unc[:, :-1].flatten(), actual_dws.t().flatten(), 1))(np.unique(causal_unc[:, :-1])), c='black')
axes[0].scatter(causal_unc[:, :-1].flatten(), actual_dws.t().flatten(), c='lightskyblue', alpha=0.5)
r_val, p_val = spearmanr(causal_unc[:, :-1].flatten(), actual_dws.t().flatten())
# axes[0].set_ylim([-5, 5])
# axes[0].text(0.18, 4, f'r={r_val:.3f}, p={p_val:.3f}')
axes[0].set_xlabel('Causal Uncertainty Before Trial')
axes[0].set_ylabel(r'$\log(\Delta dU)$')
axes[0].set_aspect('auto')

axes[1].plot(np.unique(causal_unc[:, 1:].flatten()), np.poly1d(np.polyfit(causal_unc[:, 1:].flatten(), actual_dws.t().flatten(), 1))(np.unique(causal_unc[:, 1:])), c='black')
axes[1].scatter(causal_unc[:, 1:].flatten(), actual_dws.t().flatten(), c='lightgreen', alpha=0.5)
r_val, p_val = spearmanr(causal_unc[:, 1:].flatten(), actual_dws.t().flatten())
# axes[1].text(0.16, 4, f'r={r_val:.3f}, p={p_val:.3f}')
# axes[1].set_ylim([-5, 5])
axes[1].set_xlabel('Causal Uncertainty After Trial')
axes[1].set_aspect('auto')

axes[2].plot(np.unique(delta_unc.flatten()), np.poly1d(np.polyfit(delta_unc.flatten(), actual_dws.t().flatten(), 1))(np.unique(delta_unc)), c='black')
axes[2].scatter(delta_unc.flatten(), actual_dws.t().flatten(), c='lightcoral', alpha=0.5)
r_val, p_val = spearmanr(delta_unc.flatten(), actual_dws.t().flatten())
# axes[2].text(-0.04, -2, f'r={r_val:.3f}, p={p_val:.3f}')
# axes[2].set_ylim([-5, 5])
axes[2].set_xlabel('Changes in Causal Uncertainty')
axes[2].set_aspect('auto')

# plt.plot(ms.squeeze())
# plt.show()
# plt.plot(ss.squeeze())
# plt.show()
# plt.plot(rs.squeeze())
# plt.show()
# pca by trial type 
# trial_types = all_task_types * 2 + (all_novel_outcomes+1)/2
# trial_types = trial_types.reshape(1, test_batches*batch_size).expand(hs.shape[0], hs.shape[1]).int()
# axes = vis_pca(hs.flatten(0, 1), trial_types.flatten(), \
#     ["Novel image -> Novel punishment", "Novel image -> Novel reward", "Novel image -> Non-novel punishment", "Novel image -> Non-novel reward"]);
# axes.set_title("PCA of Cell State")

# axes = vis_pca(dUs.flatten(0, 1).flatten(1, 2), trial_types.flatten(), \
#     ["Novel image -> Novel punishment", "Novel image -> Novel reward", "Novel image -> Non-novel punishment", "Novel image -> Non-novel reward"], incremental=True);
# axes.set_title("PCA of Fast Weight")

# %%

# %%

# %%

# %%
