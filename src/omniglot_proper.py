# %%
import torch
from torchvision import datasets, transforms
from modulated_full import SGRU
import torch.optim as optim
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from lamb import Lamb, get_cosine_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
from lamb import Lamb
import pickle
from decode import calculateAttention

from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine
from torchmeta.utils.data import BatchMetaDataLoader

# seed=randint(0, 100);
seed=83;
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
    global batch_size, ways, shots, img_size, device;

    # batch X (way*shot) X 1 X img_size X img_size
    train_inputs, train_targets = batch["train"]
    test_inputs, test_targets = batch["test"]

    # shuffle input order, get test image index
    perm = torch.randperm(ways*shots).to(device);
    test_idx = torch.randint(ways, size=(batch_size,)).to(device);

    # shuffle input
    train_inputs, train_targets = train_inputs.to(device)[:,perm], train_targets.to(device)[:,perm];

    # create interleaving presentation of image and label
    # on even indices, present image, present last image
    inputs_img = torch.zeros(ways*shots+1, batch_size, 1, img_size, img_size, device=device);
    test_inputs = test_inputs.to(device).transpose(0, 1)[test_idx, torch.arange(batch_size)];
    inputs_img[:-1] = train_inputs.transpose(0, 1);
    inputs_img[-1] = test_inputs;
    inputs_img = (inputs_img-inputs_img.mean()) / (inputs_img.std()+1e-8);

    # on odd indices, present label
    inputs_target = torch.ones(ways*shots+1, batch_size, device=device, dtype=torch.long)*ways;
    train_targets = train_targets.transpose(0, 1);
    inputs_target[:-1] = train_targets;

    return [inputs_img, inputs_target], test_targets.to(device).transpose(0, 1)[test_idx, torch.arange(batch_size)] # -> batch_size X 1

batch_size = 16;
ways = 20;
shots = 1;
img_size = 28;
train_batches = 1;
val_batches = 1;

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu");

# RandomAffine(degrees=11.25, translate=(0.1, 0.1)),

train_data = Omniglot("data",
                         num_classes_per_task=ways,
                         transform=Compose([Resize(img_size, interpolation=Image.LANCZOS),  ToTensor()]),
                         target_transform=Categorical(num_classes=ways),
                         class_augmentations=[Rotation([90, 180, 270])],
                         meta_train=True,
                         download=True);
train_data = ClassSplitter(train_data, shuffle=True, num_train_per_class=shots, num_test_per_class=1);

val_data = Omniglot("data",
                         num_classes_per_task=ways,
                         transform=Compose([Resize(img_size, interpolation=Image.LANCZOS), ToTensor()]),
                         target_transform=Categorical(num_classes=ways),
                         meta_val=True,
                         download=True);
val_data = ClassSplitter(val_data, shuffle=True, num_train_per_class=shots, num_test_per_class=1);
val_iter = BatchMetaDataLoader(val_data, batch_size=batch_size);

test_data = Omniglot("data",
                         num_classes_per_task=ways,
                         transform=Compose([Resize(img_size, interpolation=Image.LANCZOS), ToTensor()]),
                         target_transform=Categorical(num_classes=ways),
                         meta_test=True,
                         download=True);
test_data = ClassSplitter(test_data, shuffle=True, num_train_per_class=shots, num_test_per_class=1);
test_iter = BatchMetaDataLoader(test_data, batch_size=batch_size);


def train_eval(params):
    print(params);
    lr = params.get("lr", 1e-3);
    clip = params.get("clip", 1.0);
    clip_val = params.get("clip_val", 1.0);
    alpha_init = params.get("alpha_init", 0.01);
    tau_U_init = params.get("tau_U_init", 0.1);

    train_iter = BatchMetaDataLoader(train_data, batch_size=batch_size);
    val_iter = BatchMetaDataLoader(val_data, batch_size=batch_size);
    test_iter = BatchMetaDataLoader(test_data, batch_size=batch_size);

    model = SGRU(in_type = "image+categorical",\
                out_type = "categorical",\
                num_token = ways+1,\
                input_dim = 64,\
                hidden_dim = 512,\
                out_dim = ways,\
                num_layers = 1,\
                activation="relu",\
                mod_rank= 128,\
                reps = 4,\
                alpha_init = alpha_init, clip_val=clip_val, tau_U_init=tau_U_init\
                ).to(device);

    param_groups = add_weight_decay(model);

    optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.99), eps=1e-4);
    # scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=5e-3, factor=0.5);
    # scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.6)
    # scheduler1 = get_cosine_schedule_with_warmup(optimizer, 10000, 100000);
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000000);

    criterion = torch.nn.NLLLoss();

    loss = 0;
    trainLoss = 0;
    trainShotAcc = 0;
    last_batch = 0;
    val_errors = [];

    try:
        state_dict = torch.load("model_omniglot");
        model.load_state_dict(state_dict["model_state_dict"]);
        optimizer.load_state_dict(state_dict["optimizer_state_dict"]);
        val_errors = state_dict['val_errors'];
        last_batch = state_dict['last_batch'];print(last_batch);
        scheduler1.step(last_batch);
        print("model loaded successfully");
    except:
        print("model failed to load");

    print(model);
    print(optimizer);

    train_iter = enumerate(train_iter, start=last_batch);
    val_iter = enumerate(val_iter);
    test_iter = enumerate(test_iter);
    for idx, batch in tqdm(train_iter, position=0):
        input_total, label = offset(batch);
        new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
        new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.train().forward(\
                                                                                        x = input_total,\
                                                                                        h = new_h, \
                                                                                        v = new_v, \
                                                                                        dU = new_dU, \
                                                                                        trace = new_trace);

        loss = criterion(output[-1], label);
        trainLoss += loss.item()/100;
        trainShotAcc += torch.mean(torch.as_tensor(torch.argmax(output[-1], dim=1)==label, dtype=torch.float))/100;
        loss.backward(); #model.scale_grad();
        # for n, p in model.named_parameters():
        #     print(n, torch.norm(p.grad));
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip);
        optimizer.step();
        optimizer.zero_grad();
        scheduler1.step();
        # scheduler2.step();

        if (torch.isnan(loss)): return 0;

        if (idx+1)%100==0:
            print(trainLoss, trainShotAcc);
            trainLoss = 0;
            trainShotAcc = 0;
            if (not torch.isnan(loss)): 
                torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'val_errors': val_errors,
                    'last_batch': idx+1}, 
                    'model_omniglot');

            valLoss = 0;
            valShotAccuracy = 0;
            with torch.no_grad():
                for jdx, batch in tqdm(val_iter, position=0):
                    input_total, label = offset(batch);
                    new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
                    new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.eval().forward(\
                                                                                        x = input_total,\
                                                                                        h = new_h, \
                                                                                        v = new_v, \
                                                                                        dU = new_dU, \
                                                                                        trace = new_trace);
                    valLoss += criterion(output[-1], label)/50;
                    valShotAccuracy += torch.mean(torch.as_tensor(torch.argmax(output[-1], dim=1)==label, dtype=torch.float))/50;
                    if ((jdx+1)%50==0):
                        print(valLoss, valShotAccuracy);
                        val_errors.append(valShotAccuracy.item());
                        break;

        if (idx+1)%train_batches==0:
            print('training complete, proceeding to test')
            break;

    testLoss = 0;
    testShotAccuracy = 0;

    hs = [];
    dUs = [];
    trace_es = [];

    with torch.no_grad():
        for jdx, batch in tqdm(test_iter, position=0):
            input_total, label = offset(batch);
            new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);
            new_v, new_h, new_dU, new_trace, (last_layer_out, output) = model.eval().forward(\
                                                                                x = input_total,\
                                                                                h = new_h, \
                                                                                v = new_v, \
                                                                                dU = new_dU, \
                                                                                trace = new_trace);
            
            

            testLoss += criterion(output[-1], label)/val_batches;

            hs.append(last_layer_out['vals']);
            trace_es.append(last_layer_out['keys']);
            dUs.append(last_layer_out['dicts']);

            testShotAccuracy += torch.mean(torch.as_tensor(torch.argmax(output[-1], dim=1)==label, dtype=torch.float))/val_batches;
            if ((jdx+1)%val_batches==0):
                print(testLoss, testShotAccuracy);
                break;

    return testShotAccuracy.item(), torch.cat(hs, dim=1), torch.cat(dUs, dim=1), torch.cat(trace_es, dim=1);


acc, hs, dUs, trace_es = train_eval({});
# calculate the attention index over the previous 
t_steps = 4*(shots*ways+1);

assert(hs.shape==(t_steps, val_batches*batch_size, 512)), "hs's shape is {hs.shape}"
assert(dUs.shape==(t_steps, val_batches*batch_size, 512, 512)), "dUs's shape is {dUs.shape}"
assert(trace_es.shape==(t_steps, val_batches*batch_size, 512)), "trace_es's shape is {trace_es.shape}"
ws_h = np.empty((t_steps, val_batches*batch_size, t_steps));
ws_e = np.empty((t_steps, val_batches*batch_size, t_steps));
for i in range(1, t_steps):
    ws_i = np.stack(calculateAttention(dUs[i], hs[i-1], torch.cat([hs[:i], trace_es[:i]], dim=0)), axis=0); # --> batch_size x num_steps
    ws_i_h = np.concatenate([ws_i[:t_steps], np.empty((val_batches*batch_size, t_steps-i))], axis=1);
    ws_i_e = np.concatenate([ws_i[t_steps:], np.empty((val_batches*batch_size, t_steps-i))], axis=1);
    ws_e[i] = ws_i_e
    ws_h[i] = ws_i_h


ws = np.ma.masked_invalid(ws)

cmap = plt.get_cmap('coolwarm')
cmap.set_bad('black')

fig, axes = plt.subplots(2, 5)
rand_idx = np.random.randint(0, val_batches*batch_size, size=(10,))

xs = np.arange(0, t_steps);
ys = np.arange(0, t_steps);

for i in range(10):
    im = axes[i//5, i%5].pcolormesh(xs, ys, ws[:,rand_idx[i],:], cmap = cmap, edgecolors = None)
    fig.colorbar(im, ax=axes[i//5, i%5]);



# %%
