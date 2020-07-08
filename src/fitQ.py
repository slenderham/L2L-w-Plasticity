import torch
import numpy as np
from scipy.optimize import minimize

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

    return Qls, Qrs, nll;


# position, reward, Q of choice, diff in Q, 
# def regression_fit()
