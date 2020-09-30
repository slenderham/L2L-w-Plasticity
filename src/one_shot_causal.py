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
# from decomposition import calculateAttention, vis_parafac

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

    # batch X num_pics X 1 X img_size X img_size
    train_inputs, _ = batch["train"]
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
    # shuffle input
    train_inputs_shuff = torch.zeros(batch_size, num_pics_per_trial*num_trials, *train_inputs.shape[2:]);
    for i in range(batch_size):
        train_inputs_shuff[i] = train_inputs[[i]*num_pics_per_trial*num_trials, orders[i].long()];
    assert(train_inputs_shuff.shape==(batch_size, num_pics_per_trial*num_trials, 1, img_size, img_size))
    # separate the sequence into trials, and add blank screen after each trial
    train_inputs_shuff = train_inputs_shuff.reshape(batch_size, num_trials, num_pics_per_trial, 1, img_size, img_size);
    train_inputs_shuff = torch.cat([train_inputs_shuff, torch.zeros(batch_size, num_trials, 1, 1, img_size, img_size)], dim=2)
    train_inputs_shuff = train_inputs_shuff.flatten(1, 2)
    
    # add each presented image after entire trial to query for rating (each for a few timesteps)
    bonus_round_idx = torch.argsort(torch.rand(size=(batch_size, num_pics)), dim=-1);
    bonus_round_train_inputs = torch.zeros_like(train_inputs);
    for i in range(batch_size):
        bonus_round_train_inputs[i] = train_inputs[[i]*num_pics, bonus_round_idx[i]];
    bonus_round_novel_outcome_idx = torch.argmax(bonus_round_idx, dim=-1);

    train_inputs_shuff = torch.cat([train_inputs_shuff, bonus_round_train_inputs, torch.zeros(batch_size, num_repeats, 1, img_size, img_size)], dim=1); 
    train_inputs_shuff = train_inputs_shuff.transpose(0, 1);

    fig, axes = plt.subplots(num_trials, num_pics_per_trial+1)
    for i in range(num_trials):
        for j in range(num_pics_per_trial+1):
            axes[i][j].imshow(train_inputs_shuff[i*(num_pics_per_trial+1)+j,0,0])
    plt.show()
    fig, axes = plt.subplots(num_pics)
    for i in range(num_pics):
        axes[i].imshow(train_inputs_shuff[num_trials*(num_pics_per_trial+1)+i,0,0])
    plt.show()

    train_inputs_shuff = torch.repeat_interleave(train_inputs_shuff, repeats=num_repeats, dim=0)
    # num_repeats* (num_trials*(num_pics_per_trial+1)+num_pics+1), batch_size, 1, img_size, img_size

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
            novel_outcome_loc = [t for t in range(num_trials) if t!=novel_stim_loc[i]][torch.randperm(num_trials-1)[0]]
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

    print(outcomes_total[:,0])
    print(task_types[0])

    print(bonus_round_novel_outcome_idx[0])

    assert(train_inputs_shuff.shape[:2]==outcomes_total.shape[:2]==phase_ind.shape[:2])

    return [train_inputs_shuff.to(device), 8*torch.cat([phase_ind, outcomes_total], dim=-1).to(device)], \
                    bonus_round_novel_outcome_idx.to(device), task_types.to(device), novel_outcome.to(device);

batch_size = 64;
num_pics = 3;
len_seq = 1;
num_trials = 5;
num_pics_per_trial = 5;
num_repeats = 3;
freqs = [16, 8, 1];
assert(sum(freqs)==num_trials*num_pics_per_trial)
assert(len(freqs)==num_pics)
assert(all([freqs[i]>=freqs[i+1] for i in range(num_pics-1)]));
assert(100%len_seq==0)
img_size = 28;
train_batches = 1e6;
val_batches = 50;

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
            hidden_dim = 128,\
            out_dim = 1,\
            num_layers = 1,\
            activation="relu",\
            mod_rank= 32).to(device);

param_groups = add_weight_decay(model);

optimizer = optim.AdamW(param_groups, lr=lr);

loss = 0;
trainLoss = 0;
trainShotAcc = 0;
last_batch = 0;
val_errors = [];

try:
    state_dict = torch.load("model_causal", map_location=device);
    print(model.load_state_dict(state_dict["model_state_dict"]));
    optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
    scheduler.load_state_dict(state_dict["scheduler_State_dict"]);
    val_errors = state_dict['val_errors'];print(val_errors[-1])
    last_batch = state_dict['last_batch'];print(last_batch);
    print("model loaded successfully");
except:
    print("model failed to load");

print(model);
print(optimizer);

cumReward = []
episode_buffer = Memory()
ppo = PPO(policy = model, \
          optimizer = optimizer, \
          seq_len = len_seq,\
          buffer_size = batch_size,\
          beta_v =0.4,\
          gamma = 0.95,\
          lambd = 0.95
         );

train_iter = enumerate(train_iter, start=last_batch);
val_iter = enumerate(val_iter);
test_iter = enumerate(test_iter);

new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
for idx, batch in tqdm(train_iter, position=0):
    with torch.no_grad():
        input_total, bonus_round_novel_outcome_idx, task_types, novel_outcome = offset(batch);
        new_v, new_h, new_dU, new_trace, (last_layer_out, output, value), mod = model.train().forward(\
                                                                                x = input_total,\
                                                                                h = new_h, \
                                                                                v = new_v, \
                                                                                dU = new_dU, \
                                                                                trace = new_trace);

        # sample an action
        log_probs = output[-(num_pics+1)*num_repeats:-num_repeats:num_repeats]; # should be num_pics X batch_size X 1
        log_probs = log_probs.squeeze(-1).t();
        log_probs = torch.nn.functional.log_softmax(log_probs, -1);

        m = torch.distributions.Categorical(logits = log_probs);
        action_idx = m.sample();

        # get reward
        got_novel_trial = (torch.rand(batch_size)>0.4).to(device);
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

    # update the policy every [buffer_size] steps
    if (idx+1)%len_seq==0:
        ppo.update(episode_buffer, task='one_shot', num_pics=num_pics, num_repeats=num_repeats);
        episode_buffer.clear_memory();
        new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
    # scheduler1.step();

    if (idx+1)%100==0:
        with torch.no_grad():
            valReward = 0;
            for jdx, batch in tqdm(val_iter, position=0):
                input_total, bonus_round_novel_outcome_idx, task_types, novel_outcome = offset(batch);
                new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
                new_v, new_h, new_dU, new_trace, (last_layer_out, output, value), mod = model.train().forward(\
                                                                                        x = input_total,\
                                                                                        h = new_h, \
                                                                                        v = new_v, \
                                                                                        dU = new_dU, \
                                                                                        trace = new_trace);

                # sample an action
                log_probs = output[-(num_pics+1)*num_repeats:-num_repeats:num_repeats]; # should be num_pics X batch_size X 1
                log_probs = log_probs.squeeze(-1).t();
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
                valReward += reward.mean()/val_batches;
                if ((jdx+1)%50==0):
                    print(valReward)
                    cumReward.append(valReward);
                    torch.save({'model_state_dict': model.state_dict(), \
                                'optimizer_state_dict': optimizer.state_dict(), \
                                'cumReward': cumReward}, 'model_one_shot');
                    break;

    if (idx+1)%train_batches==0:
        print('training complete, proceeding to test')
        break;