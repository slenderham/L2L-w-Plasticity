import torch
import torch.distributions as distributions
import numpy as np
from scipy.optimize import minimize
from scipy import special
from scipy.stats import entropy
import math

def fitQ(actions, rewards):
    res = minimize(fun=lambda alphabetaeps, actions, rewards : loglikelihood(alphabetaeps, actions, rewards)[-1],
                    x0=np.array([0.5, 1, 0.1]), 
                    args=(actions, rewards),
                    bounds=((-np.inf, np.inf), (0, np.inf), (0, 1)));
    alphabetaeps = res.x; # learning rate, inverse temp, lapse rate

    Qls, Qrs, nll = loglikelihood(alphabetaeps, actions, rewards)

    return alphabetaeps, Qls, Qrs, nll;
    
def loglikelihood(alphabetaeps, actions, rewards):
    alpha = alphabetaeps[0];
    beta = alphabetaeps[1];
    eps = alphabetaeps[2];

    Qls = [];
    Qrs = [];

    nll = 0;

    for j in range(len(actions)):

        Qls.append([0]);
        Qrs.append([0]);

        for i in range(len(actions[j])):
            reward_t = rewards[j][i];
            action_t = actions[j][i];

            Qprob_l = eps + (1-2*eps)/(1+np.exp(-beta*(Qls[-1][-1]-Qrs[-1][-1])));

            if (action_t=="left"):
                nll -= np.log(Qprob_l+1e-6);
                Qls[-1].append(Qls[-1][-1] + alpha*(reward_t-Qls[-1][-1]));
                Qrs[-1].append(Qrs[-1][-1]);
            else:
                nll -= np.log(1-Qprob_l+1e-6);
                Qrs[-1].append(Qrs[-1][-1] + alpha*(reward_t-Qrs[-1][-1]));
                Qls[-1].append(Qls[-1][-1]);

    print(alphabetaeps, nll)
    return Qls, Qrs, nll;

def fitCausal(stims, outcomes, ratings, bonus_round_stims):
    best_res = None;
    best_cost = +1e6
    deltasgammatau = None
    for i in range(1):
        res = minimize(fun=lambda deltasgammatau, stims, outcomes, ratings, bonus_round_stims: ratingLikelihood(deltasgammatau, stims, outcomes, ratings, bonus_round_stims)[-1],
                    x0=[np.random.rand(1), np.random.rand(1), np.random.rand(1), 10**(np.random.rand(1)*0.3+1.8)], 
                    args=(stims, outcomes, ratings, bonus_round_stims),
                    bounds=((0, np.inf), (0, np.inf), (0, np.inf), (1, np.inf)));
        lrs, all_alphas, mse = ratingLikelihood(res.x, stims, outcomes, ratings, bonus_round_stims)
        if mse<best_cost:
            deltasgammatau = res.x; 
            best_cost = mse;

    return deltasgammatau, lrs, all_alphas, best_cost;

def ratingLikelihood(deltasgammatau, stims, outcomes, ratings, bonus_round_stims):
    delta_p = deltasgammatau[0]
    delta_r = deltasgammatau[1]
    gamma = deltasgammatau[2]
    tau = deltasgammatau[3]
    prior = 1
    prior = [prior]*3;

    batch_size, trials, im_per_trial = stims.shape
    num_pics = stims.max().int().numpy()+1
    all_learning_rates = np.zeros((batch_size, trials, num_pics))
    all_alphas = np.zeros((batch_size, trials+1, num_pics))
    assert(outcomes.shape==(batch_size, trials));
    assert(bonus_round_stims.shape==ratings.shape==(batch_size, num_pics));
    assert(stims.max()+1==len(prior));

    mse = 0
    # kl = 0

    for i in range(batch_size):
        alphas = np.array(prior).astype(float);
        for j in range(trials):
            counts = np.array([torch.sum(stims[i][j]==k).numpy() for k in range(num_pics)]);
            # print('counts', counts)
            assert(counts.sum()==im_per_trial)
            alphas = alphas*(alphas>=0);
            all_alphas[i][j] = alphas
            sum_alphas = np.sum(alphas);
            variances = (alphas*(sum_alphas-alphas))/(sum_alphas**2*(sum_alphas+1)+1e-6)
            # print('variances', variances)
            learning_rates = special.softmax(tau*variances, axis=-1)
            # print('learning rate', learning_rates)
            all_learning_rates[i][j] = learning_rates
            alphas += gamma*learning_rates*outcomes[i][j].numpy()*(counts.astype(float) + delta_p*(np.arange(num_pics)==stims[i][j][0]) + delta_r*(np.arange(num_pics)==stims[i][j][-1]));
            # print('alphas', alphas)
        alphas = alphas*(alphas>=0);
        all_alphas[i][-1] = alphas
        means = alphas/((alphas).sum()+1e-6)
        rev_ind = np.argsort(bonus_round_stims[i].numpy());
        mse += np.sum((ratings[[i]*num_pics, rev_ind].numpy()-means)**2)
        # print(ratings[[i]*num_pics, rev_ind].numpy(), means)
        # kl += entropy(ratings[[i]*num_pics, rev_ind].numpy(), means+1e-6) \
            # + entropy(means+1e-6, ratings[[i]*num_pics, rev_ind].numpy())
        # print(ratings[[i]*num_pics, rev_ind].numpy(), means)
    print(deltasgammatau, mse)

    return all_learning_rates, all_alphas, mse