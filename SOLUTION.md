# Reproducibility

To make the solution work, one has to download the file that is provided by a link, name it challenge_challenge.mat and put to the same directory where the file applicant_solution.py is located.


# Modified components

Actually, the only modofied component is cancellation algorithm with a little preparation.

### Core idea

The core idea is to use SVD decomposition after removing the estimated transmitter interference.  
So, the received signal is modeled as

$$
r = s + i_{tx} + i_{ext} + \eta
$$

First, I subtract one of the given noise components using helper functions from the provided file.  
We do it because TX leakage is the strongest structured component, so without removing it first, the decomposition mostly captures TX instead of the remaining interference.

After that, I apply SVD decomposition:

$$
X = U \Sigma V^*
$$

The key assumption is that the residual external interference is spatially coherent across channels, so it can be approximated as a rank-1 matrix:

$$
i_{ext} \approx u a^*
$$

where \(u\) is the shared temporal component and \(a\) describes channel-wise spatial weights.

Because of this low-rank structure, the first singular component should contain most of the interference energy.  
So, the first right singular vector \(v_1\) gives the dominant spatial direction of interference.

Then the shared interference is estimated as

$$
u = X v_1
$$

For each channel, projection coefficient is computed as follows:

$$
\alpha_k = \frac{u^* x_k}{u^* u}
$$

which is just least-squares projection of channel \(x_k\) onto the shared component.

Then reconstructed interference is

$$
\hat{x}_k = \alpha_k u
$$

and final output is

$$
y_k = r_k- \hat{x}_k,
$$
where $r_k$ denotes residual components.

At that moment, the metric was 6.46.

### Improvement No.1
Now, I started adapting parameters. First, I found that the coefficient $\alpha_k$, computed from the filtered signal, was not always optimal for subtraction in the original broadband domain.

Indeed, $\alpha_k$ is estimated as projection onto the shared component:

$$
\alpha_k = \frac{u^* x_k}{u^* u}
$$

which gives a good estimate of interference contribution inside the filtered subspace, but after returning to the original signal this scaling may become suboptimal.

So, after reconstructing

$$
\hat{x}_k = \alpha_k u
$$

I added an additional least-squares refinement step:

$$
\beta_k = \frac{\hat{x}_k^* x_k}{\hat{x}_k^* \hat{x}_k}
$$

This solves the minimization problem

$$
\min_{\beta} \|x_k - \beta \hat{x}_k\|^2
$$

and gives a better channel-wise scaling of the estimated interference before subtraction.

Thus, the final reconstructed interference becomes

$$
\tilde{x}_k = \beta_k \hat{x}_k
$$

and final subtraction is

$$
y_k = r_k - \tilde{x}_k
$$

#### At that point metric was $\approx 8$.

### Improvement No. 2


Since the interference is not fully stationary over time, I process the signal in local windows instead of using one global decomposition.

Window size was chosen as a tradeoff:
- larger windows give more stable SVD estimation,
- smaller windows adapt better to temporal changes in interference.

An overlap between windows was also introduced. This improves smoothness of reconstruction over time.

Additionally, after least-squares refinement, I additionally scale the reconstructed interference by a factor smaller than 1:

$$
\tilde{x}_k = \gamma \beta_k \hat{x}_k
$$

with empirical coefficient

$$
\gamma < 1
$$

(in my final version, \(\gamma = 0.72\)).

This is a conservative regularization step to avoid over-subtraction.

#### At that point metric was $\approx 8.6$.


### Improvement No. 3

At first, I used only the first singular component, which corresponds to the standard rank-1 approximation.

I assumed that the residual interference was not always perfectly rank-1. I called it rank-1.5.

In some windows, the second singular value was also significant, which suggests that part of the structured interference is not fully captured by the dominant mode.

So I extended the model from

$
i_{ext} \approx u_1 a_1^*
$

to a more flexible low-rank approximation:

$$
i_{ext} \approx u_1 a_1^* + u_2 a_2^*
$$

where the first term captures the dominant coherent interference, and the second term captures weaker residual structure.

Thus, instead of using only

$$
u_1 = X v_1
$$

I also estimate

$$
u_2 = X v_2
$$

using the second right singular vector.

For each mode, projection and least-squares refinement are applied independently, similarly to the rank-1 case.

The final cancellation is computed as a weighted combination:

$$
\hat{x}_k =  k_1 \beta_1 \hat{x}_{k,1} + k_2 \beta_2 \hat{x}_{k,2}
$$

where $k_i$ are adaptive coefficients.

To choose these coefficients, I use the singular value ratio ()

$$
d = \frac{\sigma_1}{\sigma_2}
$$

which measures how close the local interference is to rank-1.

If d is large, then the first mode is clearly dominant, so stronger subtraction is applied to the first component and very weak subtraction to the second.

If \(d\) is smaller, the interference is less coherent and closer to rank-2, so the first subtraction is made less and the second mode is allowed slightly larger.
Next obstacle was to properly find $k_i$. From using if-else blocks and standard function, the best profit was achieved at a very weird function
$$k_1 = \theta_1 + \theta_2 \cdot \tanh(\frac{d - 3}{4})$$ with thetas are hyperparameters. This was obtained suddenly, but the motivation is described, so we had to only choose a function that does so, and this gave the best performance.
Meanwhile, for $k_2$ such function was not found, and it uses if-else blocks.
#### At this point, the metric was about 9.9

### Improvement No. 4

As a final safety step, as always, I compare filtered power before and after cancellation.  
If the residual power does not decrease sufficiently, I apply a small shrinkage to the cancellation term, which usually improves stability. 

## Final approach and metric

#### So the final method is a combination of ideas listed above. The final metric achieved after tuning all the parameters was 10.10 


## Experiments and failed attempts

### Other core ideas

1. **ICA (Independent Component Analysis)**  
Although the final pipeline is structurally similar to ICA, a direct ICA use shown worse metrics.

2. **FFT / Fourier-domain approaches**

A number of Fourier-based modifications were tested, but they did not consistently improve the metric.

In particular, the following variants were tested:

- performing SVD on full FFT matrix
- selecting only high-energy frequency bins
- frequency-wise scaling:
  $$
  \beta_k(f) = \frac{\hat X^*(f)X(f)}{|\hat X(f)|^2}
  $$

However, these approaches did not give much: the score filter already performs a band-limiting operation, making additional Fourier separation not as an improvement but as an alternative.
Fourier methods often failed explainability even after tuning parameters.
Moreover, Fourier transforms increased model complexity without providing a stable gain in the evaluation metric and compromising future development due to its complexity, so they were rejected.


3. **PCA vs SVD**

PCA was considered as an alternative formulation:

However, in practice SVD was more efficient as directly provided $v_i$ and projections $Xv_i$.


### Improvements of the existing algorithm that did not work

1. **Higher-rank extensions (3+ components)**

Extending the model from rank-2 to rank-3:

$$
X \approx u_1 v_1^* + u_2 v_2^* + u_3 v_3^*
$$

did not improve the metric.

I think that is because the third singular component typically corresponds to either noise subspace or weak residual structure with low coherence.

2. **Whitening before SVD**

I applied channel whitening before SVD by normalizing the data with the inverse square root of the empirical channel covariance matrix.

But on the final implementation, this reduced performance (10.10 → 7.39), likely because the original channel power differences were informative. This also can be intuitevely seen as 1.5-rank decomposition was better #for interference strength estimation and were partially destroyed by whitening.