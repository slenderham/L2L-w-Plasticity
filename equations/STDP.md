- RNN dynamics

- $$
  v_t = (1-z) * v_{t-1} + z * (Wx_{t-1} + (U + \alpha\otimes dU)h_{t-1} + b)\\
  h_t = [v_t]_+\\
  z = \sigma(Wx + Uh + b)
  $$

- STDP

  $$
    \tau_+ \dot{e}_j = -e_j + \sum h_{j}\\
    \tau_- \dot{e}_i = -e_i + \sum h_{i}\\
    \dot{w}_{ij} = \alpha_+ e_{j}h_i - \alpha_- e_i h_j
  $$

  - in discrete time

$$
    e_{t,j} = (1-\tau_+ )e_{t-1,j} + \tau_+ h_{t,j}\\
    e_{t,i} = (1-\tau_- )e_{t-1,i} + \tau_- h_{t,i}\\
    {dU}_{t,ij} = (1-\tau_U)dU_{t-1,ij} + \tau_U(\alpha_+ e_{t,j}h_{t,i} - \alpha_- e_{t,i} h_{t,j})
$$

  - if modulated 

    $$
    e_{t} = (1-\tau_e )e_{t-1} + \tau_e h_{t}\\
      {E}_{t} = (1-\tau_E)E_{t-1} + \tau_E(h_{t}e_t^T - e_{t} h_{t}^T)\\
      dU_{t} = (1-\tau_U)dU_{t-1} + \tau_UM_{t}E_{t}
    $$
  
- computation steps

  - v, h, dU from previous time step
  - calculate new v, new h
  - use both to update dU
  - return all three

- Why STDP works? following hinton's fast weights papers

  - assuming that a+ = a-, tau+ = tau-

    $$
    U^{\text{fast}}_Tx = \sum_{t=1}^T \tau_E(1-\tau_E)^{T-t}(h_te_{t-1}^\top - e_{t-2}h_{t-1}^\top)x\\
    = \sum_{t=1}^T \tau_E(1-\tau_E)^{T-t}h_te_{t-1}^\top x - \sum_{t=1}^{T-1} \tau_E(1-\tau_E)^{T-t}e_{t-1}h_{t}^\top x\\
     = \sum_{t=1}^T Diag(\tau_{E,t})h_{t} \sum_{s=1}^{t-1}[Diag(\tau_{e,s})h_s]^\top x - \sum_{t=1}^T \tau_E(1-\tau_E)^{T-t}\sum_{s=1}^{t-1}(Diag(\tau_{e,s}) h_s)h_{t}^\top x
    $$
    - 
    - first term is the "attending to the recent past term", where each past experience is weighted by the resemblemce of its history with the current input

    $$
        \sum_{t=1}^T \bigg[\sum_{s=1}^{t-1}(\tau_E(1-\tau_E)^{T-t}h_s^\top (\tau_e(1-\tau_e)^{t-s})x \bigg] h_t
    $$

    - second term

      $$
        \sum_{t=1}^T \tau_E(1-\tau_E)^{T-t}\sum_{s=1}^{t-1}Diag(\tau_{e,s})h_sh_{t}^\top x \\
        = \sum_{t=1}^T\sum_{s=1}^{t-1} \tau_E(1-\tau_E)^{T-t}Diag(\tau_{e,s})h_sh_{t}^\top x\\
        = \sum_{s=1}^T\sum_{t=s+1}^T \tau_E(1-\tau_E)^{T-t}Diag(\tau_{e,s})h_sh_{t}^\top x\\
        = \sum_{t=1}^T\bigg[\sum_{s=t+1}^{T} \tau_E(1-\tau_E)^{T-s}Diag(\tau_{e,t})h_{s}^\top x\bigg]h_{t}
      $$
      
      - 

    - More generally with modulation

    - the weight update

       $$
       h = w^Tz\\
        \frac{dw}{dt} = h<x>-<h>x\\
        <\frac{dw}{dt}> = <w^Tx<x>-w^T<x>x> = 0\\
        \frac{d||w||^2}{dt} = 2w^T\frac{dw}{dt} = 2(w^Tx)w^T<x>-2w^T(w^T<x>)x=0\\
    	 h_t = w\top h_{t-1}\\
    	 \frac{dw}{dt} = h_t<h_{t-1}>-<h_t>h_{t-1}\\
    	  <\frac{dw}{dt}> = <w\top h_{t-1}<x>-w^T<x>x> = 0\\
    	$$

  

- tasks:
  
  - language modelling (char level)
    - model doesn't use fast weight!!!! limiting factors
      - clip value
      - alpha
      - tau_U
  - **associative recall with omniglot**
  - **wisconsin card sort**
  - make continuous time RNN, no adaptive time constant
    - Sequential MNIST with Delay Noise: both seem to be doing fine
  - meta learning time series transformation
  - add fixation output to force the network activity level to remain constant
  - **two forced choice maze task**
  - able to procedurally generate, randomize 
  
- analysis:

  - **tensor decomposition on the neural activity:**
    - noise: stimulus
    - dimension: exposure, chunk, channel
  - linear SVM decoding of stimulus and task relevant information
    - bilinear decoding for the weight matrix?
    - decoding using the eigenvectors of the weight matrix
  - jPCA of the weight matrix, for obtaining rotation
  - activity disruption
    
    - add noise to activity layer
  - **turn off alpha during testing**
  - activation correlation plot
  - overlap analysis of fast weight and activities
  - pca of mod activities
  - **gradient analysis!!! (or rather sensitivity analysis) on ART**
  - **turn off plasticity, analyze fixed point**
  - **tensor decomposition of**
  - **PCA of modulation activity**
  - **modulatory neurons** - what do they calculate: visualize weights
    - RPE = r - Q
    - Q_choice = Q
  - estimating Q:
  
      $$
        Q_t(a) = Q_{t-1}(a) + \alpha(r_{t-1}-Q_{t-1}(a)) \\
        \hat{P}_t(L) = \frac{\exp(-\beta Q_t(L))}{\exp(-\beta Q_t(L))+\exp(-\beta Q_t(R))}\\\ \\
        = \frac{1}{1+\exp(-\beta(Q_t(R)-Q_t(L)))}
    $$
  
    - we have $P_t(L)$
    - minimize 
    
    $$
        KL(P_t||\hat{P}_t) = - P_t(L)\ln \hat{P}_t(L) - P_t(R)\ln \hat{P}_t(R)
  $$
  
  - "premembering": duality of memory and learning

|                        | Memory-Augmented | Non-Memory Augmented |
| :--------------------- | ---------------- | -------------------- |
| Gradient Following     |                  |                      |
| Not Gradient Following |                  |                      |

| Neuromodulator                                               | Biological Function                     | Corresponding Gate Variable | Parameters of MDP       |
| ------------------------------------------------------------ | --------------------------------------- | --------------------------- | ----------------------- |
| Aceytocholine, Noradrenaline, Dopamine, Seretonin (Brzosko et al., 2013). Specifically, Aceytocholine ~ learning rate, Dopamine ~ reward prediction error (Doya, 2002). | Polarity/Sign of STDP                   | $M$                         | Reward Prediction Error |
| Aceytocholine, Noradrenaline, Dopamine (Brzosko et al., 2013), Seretonin ~ discount factor (Doya, 2002) | Induction of STDP                       | $s$                         | Discount Factor         |
| "Candidates of detectors of presynaptic events are, for example, the amount of glutamate bound ([Karmarkar and Buonomano, 2002](https://www.jneurosci.org/content/26/38/9673#ref-18)) or the number of NMDA receptors in an activated state ([Senn et al., 2001](https://www.jneurosci.org/content/26/38/9673#ref-29)). Postsynaptic detectors *o*1 and *o*2 could represent the influx of calcium concentration through voltage-gated Ca2+ channels and NMDA channels ([Karmarkar and Buonomano, 2002](https://www.jneurosci.org/content/26/38/9673#ref-18)) or the number of secondary messengers in a deactivated state of the NMDA receptor ([Senn et al., 2001](https://www.jneurosci.org/content/26/38/9673#ref-29)) or the voltage trace of a back-propagating action potential ([Shouval et al., 2002](https://www.jneurosci.org/content/26/38/9673#ref-30))." (Pfister & Gerstner, 2006). Modulated by Noradrenaline, Dopamine (Brzosko et al., 2013). | Neuronal Traces, Control of STDP Window | $r$                         | --                      |

- Brzosko et al., 2013. Neuromodulation of Spike-Timing-Dependent Plasticity: Past, Present, and Future
- Pfister & Gerstner, 2006. Triplets of Spikes in a Model of Spike Timing-Dependent Plasticity
- Doya, 2002. Metalearning and neuromodulation

- 