- Network Equations (updated):

  $$
  z_t = \sigma(W_zx_t+U_zh_{t-1}+b_z) \in \R^n \\
  v_t = (1-z_t)v_{t-1}+z_t(W_vx+U_vh_{t-1}+\alpha U^{fast}_{t-1}h_{t-1}+b_v) \in \R^n\\
  h_t = [v_t]_+ \in \R^n\\
  mod_t = [W_{mod}h_t+b_{mod}]_+ \in \R^m, m<n
  $$

    - The modulation signals ($m_t, s_t$) and learning rates ($\alpha, \tau_U$) are made scalar, which makes the derivation in the previous report valid, and also saves parameter. No drop in performance was found. This would make the fast weight perfectly skew-symmetric. The eligibility traces and fast weights are updated similarly to the last write-up.

  $$
  \text{neuronal eligibility trace, related to the time window of STDP, controlled by r-gate}\\
  r_t = \sigma(w_r^\top mod_t+b_r) \in \R\\
  e_t = (1-r_t)e_{t-1}+r_th_{t} \in \R^n \\\ \\
  \text{synaptic eligibility trace, controlled by s-gate}\\
  s_t = \sigma(w_s^\top mod_t+b_s) \in \R\\
  E_t = (1-s_t)E_{t-1}+s_t(h_te_{t-1}+e_{t-1}h_t) \in \R^{n\times n}\\\ \\
  \text{fast weight, controlled by m, which can be positive or negative}\\
  m_t = w_m^\top mod_t+b_m \in \R\\
  U^{fast}_t = (1-\tau_U)U^{fast}_{t-1}+\tau_Um_tE_t \in \R^{n\times n}
  $$

    - alpha is the softplus transformation of a free parameter $\alpha = \ln(1+\exp(\tilde\alpha))\in [0,\infty)$. tau_U is the sigmoid transformation of another free parameter $\tau_U = \sigma(\tilde\tau_U) \in [0,1]$. Initializations of these parameters matter a lot.
    - The fast weight's magnitude is contrained otherwise the network's activity can be unstable. We want $|U_v+\alpha U^{fast}|<C$. So we enforce each entry of the fast weight to be $(-C-U_v)/\alpha<U^{fast}<(C-U_v)/\alpha$

- Omniglot Few Shot Classification
  
    - In each episode, the network is given a set of support images, their corresponding labels, and a test image. The task is to provide the label of the test image, which should be of the same category with one or more of the support images.
    - The is a hard version of the Associative Retrieval Task: the network needs to learn the association between image and label, and retrieve the correct label. 
    - The image input to the network is encoded by a four layer CNN (3x3 conv->2x2 max pool->ReLU->BatchNorm). The label is encoded with a embedding layer. They are concatenated and passed to the recurrent network as input.
    - Each pair is presented at least 2 times (**presenting more times actually lead to faster convergence in experiments I tried, as long as $\tau_U$ is initialized properly to account for longer sequence length**). Then the test image (only one) is presented for a few times, concatenated with a novel label (for example, if there are 20 categories in each episode, the test image is concatenated with an embedding of 21).
    
- Causal Learning

    - The task was used to test a learner's ability to control the speed at which it learns causal relationships in an environment with varying causal uncertainties. In order to vary the causal uncertainty, two task types were introduced: (a) the novel stimuli is followed by the novel outcome, and (b) non-novel stimuli are followed by the novel outcome. To prevent the model from identifying one type of outcome with the novel outcome, we randomly sample the novel outcome to be the positive or negative one. This leads to a total of four combinations from stimuli type X outcome type.

    - The task is divided into multiple rounds. In each round, three distinct images are sampled. Each image will be presented different number of times during that round - two images (non-novel stimuli) will be presented more than once, and one image (novel stimuli) will be presented with only once. Each round is divided into three phases. Phase one is divided into five trials. In each trial, the network is presented with sees five images, and then one scalar value indicating the outcome of that trial. In phase two, the network is presented again with the three images in random order. We use the network's output when presented with an image as its rating for that image, which will determine the probability that that image is selected at the end of phase two. At the end of phase two, we randomly sample three trials from phase one in that round, and randomly sample one image that appears in those rounds according to the ratings given by the model. The outcome that is followed by the stimuli type of that image is given as the final reward to the model. For the current setting, the phase one of each round has five trials. 

    - The input to the network consists of three parts concatenated together: (1) the image, (2) the outcome, and (3) indicator of phase. The image input to the network is encoded by a four layer CNN (3x3 conv->2x2 max pool->ReLU->BatchNorm). One additional dimension is used to indicate the outcome of each trial, which is not 0 only when the outcome is presented. The indicator of phase are 4 dimensional one-hot vectors. The four dimensions correspond to (1) phase one stimuli presentation, (2) phase one outcome presentation, (3) phase two, (4) value function estimation, which is used for policy gradient training of the model. The network is trained with PPO.

    - Model fitting: the stimuli and ratings of the model are used to fit a Bayesian model to measure the causal uncertainties in each round. We assume the model tries to estimate how much each stimuli causes the good outcome, which is motivated by the way reward is calculated. 

    - We assume model maintains a probabilistic model of whether each type of stimuli causes the good outcome. The model is a Dirichlet distribution $\text{Dir}(\alpha_1, \alpha_2, \alpha_3)$, whose parameters are updated when new stimuli and outcomes are received. 

    - Following [[]], we fit the model with four free parameters: $\gamma_0$ the base learning rate, $\tau$ the inverse temperature of the softmax calculation, $\delta_p$ the parameter accounting for the primacy effect, and $\delta_r$ accounting for the recency effect.

    - The alpha's are updated at the end of each trial in the following way given outcome $r\in\{-1, +1\}$, the number of times each stimuli occured in that trial $x_i$ and $D(i)$ is the subset of $\{1,2,3,4,5\}$ which is are the places where stimuli $i$ occured, for $i=1,2,3$
        $$
        \alpha_{i,t+1} = [\alpha_{i,t}+\gamma_0\gamma_{i,t}r(x_i+\delta_p\mathbb{I}[1\in D(i)]+\delta_r\mathbb{I}[5\in D(i)])]_+
        $$

        - The causal strength is defined as the mean of the Dirichlet distribution $\mathbb{E}[\theta_i]=\frac{\alpha_i}{\sum_{j=1}^3\alpha_j}$. 

        - The causal uncertainty is defined as the variance of the Dirichlet distribution $\mathbb{V}[\theta_i]=\frac{\mathbb{E}[\theta_i](\sum_{j=1}^3\alpha_j-\mathbb{E}[\theta_i])}{1+\sum_{j=1}^3\alpha_j}$

        - Here $\gamma_i = \frac{\exp(\tau\mathbb{V}[\theta_i])}{\sum_i\exp(\tau\mathbb{V}[\theta_j])} \in [0, 1]$ modulates the learning rate of the model based on the causal uncertainty. The greater the uncertainty, the larger the learning rate needed to resolve the uncertainty.

        - We want the causal strengths to be close to the actual ratings given by the model. We use $\mathbb{E}[\theta_i]$ as the causal strengths. Given the ratings of the model $\widetilde{o_i}$ for each stimuli, we normalize them with a softmax function to get the normalized ratings $o_i = \frac{\exp(\beta\widetilde{o_i})}{\sum_{j=1}^3\exp(\beta\widetilde{o_j})}$. We then want 

        - $$
            \gamma_0, \tau, \delta_p, \delta_r = \arg\min (\mathbb{E}[\theta_{i,5}]-O)^2
            $$

        - $\beta$ is used to smooth the ratings of the model, since an optimized model might produce very sharp ratings that cannot be fitted readily with the current model. For the current model, $\beta$ is chosen to be $1/20$ to roughly match the ratings produced by human participants 

- Maze with switching reward

    - A small T-maze like the one below (1 are walls, 0 are legal positions)

    $$
    1\ 1\ 1\ 1\ 1\ 1\ 1 \\
    1\ 0\ l\ b\ r\ 0\ 1 \\
    1\ 0\ 1\ 0\ 1\ 0\ 1 \\
    1\ 0\ 1\ x\ 1\ 0\ 1 \\
    1\ 0\ 0\ d\ 0\ 0\ 1 \\
    1\ 1\ 1\ 1\ 1\ 1\ 1 \\
    $$

    - Episode Structure
        - Two phases
            - In the Approach phase, b=0, d=1. The agent starts at $x$. It can only move forward. After it arrives at point marked with $b$, it either moves left (to $l$) or right (to $r$). Only one will be rewarded. After it leaves $b$, it moves into the next phase.
            - In the Return phase, b=1, d=0. The agent should return back to $x$. Then the next trial starts.
        - The agent receives a negative reward for running into wall. It also receives a negative reward at each time step to encourage shorter paths.
        - The agent receives a large positive reward for arriving at the correct goal. In each trial, the reward has a probability $p$ of being at $l$, and $1-p$ at $r$. This reward probability remains the same for a few trials, then it switches. 
        - The best strategy is to constantly visit the site which leads to more reward, until a switch happens. 

    - Model Fitting 
    
        - adapted from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4588166/
        
        - To see if the model is performing implicit reinforcement learning, we fit a simple RL algorithm to the model's behavior
        - Three parameters to fit
            - $\alpha$: learning rate
            - $\beta$: inverse temperature of the softmax
            - $\epsilon$: "lapse rate", stochasticity in the model's choice
        - $Q : \{L, R\} \to \R$ is a value function that maps action (going left or right) to a real value
        - Q is updated in the following equation, as in the Rescorla-Wagner model

        $$
        Q_t(a) = Q_{t-1}(a) + \alpha(r_{t-1}-Q_{t-1}(a))
        $$

        - Action is selected with the probabilities:

        $$
        P_t(L) = \epsilon+(1-2\epsilon)\cdot \frac{\exp(-\beta Q_t(L))}{\exp(-\beta Q_t(L))+\exp(-\beta Q_t(R))}\\\ \\
        = \epsilon+\frac{1-2\epsilon}{1+\exp(-\beta(Q_t(R)-Q_t(L)))}\\\ \\
        P_t(R) = 1-P_t(L)
        $$

        - Given a set of choices made by the network $C_t, \forall t$, we can fit the parameters using maximum likelihood

        $$
        (\alpha, \beta,\epsilon) = \arg\max \sum_t \log P_t(C_t)
        $$

