import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

np.random.seed(0);



def sequentialize(batch, noise_dur, noise_scale, grace_period):
    '''
        given a batch of N * C * H * W,
        transform into a sequence of size H+(random noise) * N * W

    '''
    img, label = batch;


    img = img.squeeze().flatten(start_dim=1).t().unsqueeze(2);
#     img = torch.cat((\
#                         img, \
#                         torch.randn((noise_dur, img.shape[1], img.shape[2]))*noise_scale, \
#                         torch.zeros((grace_period, img.shape[1], img.shape[2]))),\
#                     dim=0);

    return img.to(device), label.to(device);

def equalSamples(dataset, num_classes = 10, examples_per_class = 1000, sample_size = 30):
    chosen = [];
    for i in range(num_classes):
        indices = np.random.choice(torch.where(dataset.targets==i)[0].tolist(), size=sample_size, replace=False).tolist();
        chosen += indices;

    dataset.data = dataset.data[chosen];
    dataset.targets = dataset.targets[chosen];

def decoder(x1, x2, labels, fold=4):
    num_steps, batch_size, num_channels = x1.shape;

    shuff = np.random.choice(np.arange(batch_size), size=batch_size, replace=False);
    block = batch_size//fold;

    score1 = [];
    score2 = [];

    for j in tqdm(range(num_steps)):
        x1_t = x1[j];
        x2_t = x2[j];
        score1.append([]);
        score2.append([]);
        for i in range(fold):
            model1 = linear_model.SGDClassifier(max_iter=1000, tol=1e-3);
            model2 = LinearSVC(random_state=0, tol=1e-4, dual=False);

            train_idx = np.concatenate((shuff[:i*block], shuff[(i+1)*block:]));
            test_idx = shuff[i*block:(i+1)*block];

            model1.fit(x1_t[train_idx] , labels[train_idx]);
            model2.fit(x2_t[train_idx] , labels[train_idx]);

            score1[-1].append(model1.score(x1_t[test_idx], labels[test_idx]));
            score2[-1].append(model2.score(x2_t[test_idx], labels[test_idx]));


    print(score1, score2);

def calculateAttention(mem, query, history):
    value = np.matmul(mem, np.expand_dims(query, -1)).squeeze();
    # value should be a linear combination of previous states -  Hw = Mq, try to find w
    # Do linear regression! w = (H'H)^-1H'Mq
    w = [];
    print(history.shape, value.shape)
    for i in range(mem.shape[0]):
        w.append(LinearRegression(fit_intercept=False).fit(history[i].T, np.expand_dims(value[i],-1)).coef_);

    return w;

def calculateCCA(X):
    t_total = X.shape[0];
    corr = np.eye(t_total);
    for i in tqdm(range(t_total)):
        for j in range(i):
            cca = CCA(max_iter=100).fit(X[i], X[j]);
            R_sq = cca.score(X[i], X[j]);
            corr[i,j] = corr[j,i] = R_sq
    return corr;

# if __name__=="__main__":

#     state_dict = torch.load('L2L-w-STDP/src/model_SMNIST-RNN-large', map_location=torch.device('cpu'))
#     model = SGRU(in_type = "continuous",\
#          out_type = "categorical",\
#          num_token = 0,\
#          input_dim = 1,\
#          hidden_dim = 128,\
#          out_dim = 10,\
#          num_layers = 1,\
#          activation="relu",\
#          mod_rank = 32,\
#          );print(model);
#     model.load_state_dict(state_dict['model_state_dict']);

#     n_epochs = 12;
#     batch_size = 1;
#     drawInt = 100;
#     noise_dur = 10;
#     grace_period = 10;
#     noise_scale = 1;
#     sample_size = 100;

#     mnist_data_test = datasets.MNIST(root="L2L-w-STDP/src/data", train=False, transform=transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.1307,), (0.3081,)),
#                    ]), download = True);

#     equalSamples(mnist_data_test, sample_size=sample_size);

#     test_iter = torch.utils.data.DataLoader(, batch_size=batch_size, shuffle=True)

#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#         model.to(device);
#     else:
#         device = torch.device("cpu");


#     vs = [];
#     dUs = [];
#     outputs = [];
#     labels = [];

#     classification_error = 0;
#     fixation_error = 0;

#     with torch.no_grad():
#         for idx, batch in enumerate(tqdm(test_iter)):
#             outputs.append([]);
#             new_h, new_v, new_dU, new_trace = model.get_init_states(batch_size=batch_size, device=device);

#             inputs, targets = sequentialize(batch, noise_dur, noise_scale, grace_period);
#             labels.append(targets);

#             vs_single = [];
#             dUs_single = [];

#             for jdx in range(inputs.shape[0]):

#                 new_v, new_h, new_dU, new_trace, (_, output) = model.eval().forward(\
#                                       x = inputs[jdx:jdx+1,:,:],\
#                                       h = new_h, \
#                                       v = new_v, \
#                                       dU = new_dU, \
#                                       trace = new_trace);

#                 vs_single.append(new_v[0].squeeze());
#                 dUs_single.append(new_dU[0].squeeze());

#             vs.append(torch.stack(vs_single));
#             dUs.append(torch.stack(dUs_single));

#             classification_error += 1.0*torch.sum(torch.argmax(output.squeeze(), dim=1)!=label)/10/examples_per_class;

#     vs = torch.cat(vs, dim=1);
#     dUs = torch.cat(dUs, dim=1);
#     labels = torch.cat(labels);

#     print(classification_error);

#     torch.save([vs, dUs, labels]);

#     num_neurons = vs.shape[2];
#     vals, vecs = np.linalg.eig((model.rnns[0].h2h.weight[3:num_neurons,:].detach().numpy()+model.rnns[0].alpha.abs().detach()*dUs[0][i].detach()).squeeze());
#     decoder(vs, vecs[:,0], labels);