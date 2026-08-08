# KLQ: Geometry-aware LLM quantization, per-direction bit allocation priced by measured KL divergence given injected perturbation.
------
## Introduction
Several investigations into the geometry of LLM embedding spaces have found their activation space to be extremely anisotropic: a few fixed features consistently spike in magnitude. I confirm this in the eigendecomposition: past a certain layer, one direction dominates the covariance. This makes uniform quantization fragile, as it spends its resources evenly in a naturally uneven space. Random-rotation-based quantizers (QuaRot, QuIP#, SpinQuant…) try to fix this by making the space truly even before quantizing. In contrast, KLQ takes advantage of the model's geometry. For every weight matrix, it pinpoints the important directions by eigendecomposing the covariance matrix of activation samples from a calibration set, then perturbing activations along each direction and measuring the KL divergence from the original model's output distribution to the perturbed one. A causal measure of how important each direction is. KLQ then assigns bits per direction by water-filling, the provably optimal allocation under this model. Prior methods that used PCA to identify important directions either sorted them by variance and applied two-tier quantization regimes (CoQuant) or used this information to flatten the directions further (ResQ, SpinQuant...)

-----
## Main results
The per-direction quantizer is an interchangeable backend, and this repo ships two deliberately simple ones (uniform-grid RTN and a two-book additive VQ). Stronger codes such as lattices (QuIP#, NestQuant), trellises (QTIP), learned codebooks (AQLM, VPTQ) are drop-in replacements on this axis and orthogonal to our claim; likewise, learned rotations (SpinQuant, OSTQuant) and error-compensating rounding (GPTQ, LDLQ) address the coordinate and rounding axes respectively, not the allocation axis KLQ studies. KLQ remains training-free at the cost of one forward pass per direction per matrix per layer.
KLQ-RTN and KLQ-VQ both use FP16 64-sized grouping (taken into account for bpw calculations) and differ in grid-assignment method, RTN uses a uniform grid while VQ uses a additive vector quantization with d=8 and 256 levels. I discovered GPTQ actually hurts KLQ's performance in respect to RTN, I hypothesize this happens because GPTQ actually fights against the water-filling allocation by redistributing error into Hessian-low directions (and thus directions with little variance) which don't always correlate with KL-low directions.

|             |   Effective W/A/KV   |  Qwen 2.5 0.5B (Wikitext-2 PPL)   |      Source                                        |
|-------------|------------|-----------------------------------|----------------------------------------------------|
| **FP16**    | 16/16/16   |   *13.07*                         |   Own measure matchesCoQuant's and ResQ's number   |
| RTN         | 4/4/4      |     23204.3                       |                                                    |
| QuaRot    | 4/4/4    |     204.10                        |  CoQuant's rerun (consistent w/ ResQ's 219.9)    |
|  QUIK       | 4/4/4    |     38.6                        |    ResQ paper                                      |
| **KLQ-RTN** | 4/4/4      |     **21.07**                     |                                                    |
| **KLQ-RTN** | 4/16/16    |     **15.9**                      |                                                    |
|  ResQ       | 4.5/4.5/4.5|   18.19                         |    CoQuant rerun                                   |
| CoQuant   | 4.5/4.5/4.5|   17.76                         |    CoQuant                                         |
| **KLQ-RTN**     | 4.5/4.5/4.5|     **16.33**                 |                                                    |
| **KLQ-RTN**     | 4.5/16/16|     **14.34**                   |                                                    |

|             |   Effective W/A/KV   |  Llama 3.2 1B (Wikitext-2 PPL)    |      Source                                        |
|-------------|------------|-----------------------------------|----------------------------------------------------|
| **FP16**    | 16/16/16   |   *9.75*                          |                                                    |
|  QuaRot     | 4/4/4    |     14.59                         |    ReSpinQuant's paper                                   |
|  SpinQuant  | 4/4/4    |     13.52                          | SpinQuant's paper                               |
|  ReSpinQuant | 4/4/4     |     **13.09**                     |      ReSpinQuant's paper                         |
| **KLQ-VQ**      | 4/4/4      |     13.36                     |                                                    |
| **KLQ-VQ**      | 4/16/16      |     11.56           |                                                            |
|  QUIK       | 4.5/4.5/4.5|     21.8                          |    ResQ paper                                      |
|  ResQ       | 4.5/4.5/4.5|     12.4                        |    ResQ paper                                      |
|  CoQuant    | 4.5/4.5/4.5|     11.6                          |    CoQuant paper                                   |
| **KLQ-VQ**      | 4.5/4.5/4.5      |     **11.45**           |                                                    |
| **KLQ-VQ**      | 4.5/16/16      |     **10.52**           |                                                      |

The whole Qwen 2.5 0.5B pipeline, including the sampling from each layer, the eigenvalue decomposition, the KL damage measurements using 4 windows and 512 tokens per window, took ∼8 hours to compute on an RTX 3090.
This same pipeline for Llama 3.2 1B took ∼15 hours on the same hardware.




----
## Methods
Given a transformer model $T$ with $N$ blocks $B_i$, then $T = B_N \circ ... \circ B_1$, each $B_i$ consists of an attention block $A_i$ and an MLP $M_i$ (up to the embedding and unembedding matrices which we leave at FP16). Every learnable weight in the model belongs to exactly one block which we index by $W_{i,m}$ with $i\in\{1,\dots,N\}$ and $m\in \{ q,k,v,o,down,up,gate \}$ each corresponding to their respective part of the block. 

Each matrix $W_{i,m}$ has an input $x_{i,m}\in\mathbb{R}^{d_m}$. While calibrating we take $n$ random vectors (in our experiments $n$ equals 128 windows of 1024 sequence length so $n=131072$) with a distribution given by the calibration data and capture them to build the matrix $X_{i,m} \in R^{n \times d_m}$, with mean $\mu_{i,m}$. We compute the second moment $S_{i,m} = \frac{1}{n} X_{i,m}^\top X_{i,m}$ and then the covariance $C_{i,m} = S_{i,m} - \mu_{i,m}\mu_{i,m}^\top$. Lastly from this we get the eigendecomposition $C_{i,m} = V\Lambda V^\top$. A direction then is an eigenvector of $C_{i,m}$ so a column of $V_{i,m}$. Now if $\Lambda = \text{diag}(\lambda_1,...,\lambda_{d_m})$. Effectively this decomposes vectors as $x = \mu + \Sigma_i{z_i v_i}$, where $x$ chosen from the original distribution will have a variance of $\lambda_i$ across the value $z_i$ for the value of the direction across $v_i$ (so it will have a standard deviation of $\sigma_i = \sqrt{\lambda_i}$).

In a transformer model $T$, after we've pinpointed the directions $v_i$ with $i\in\{1, ..., m\}$ and their variance $\lambda_i$ in a specific matrix $W$ of a specific layer we proceed to measure the KL divergence of perturbing this direction. Let's consider the model $\tilde{T_i}$ in which the matrix $WV$ takes the input $V^\top (x-\mu)$, we disturb the activation $V^\top (x-\mu)$ by adding $\delta_i = \sqrt{\lambda_i}$ (one standard deviation perturbation towards the ith direction) to the ith coordinate. We deliberately use this notation to show the parallel with rotation matrices, this method intentionally uses an uneven coordinate system.

Given this disturbed model $\tilde{T}\_i$ we compute $KL_i = KL(P_T(·|D)||P_{\tilde{T}_i}(·|D))$ given data $D$ disjoint from original calibration distribution. The damage a direction suffers under a quantization of $b_i$ bits is defined as the expected KL between the original model and the quantized model. We approximate this expected value of KL with a probe as $KL_i$ using the unit-relative-error perturbation seen earlier, according to the high rate law the damage is proportional to $2^{-2b_i}$ so $\text{damage}_i(b_i) = p_i 2^{-2b_i}$ where $p_i$ will be a price function. 

The price then will be $p_i = ||Wv_i||^2 KL_i$. The $KL_i$ is augmented by $||Wv_i||^2$, a read-energy factor, which gives a consistent 0.1-0.4ppl improvement over pure KL. 

If we treat each direction as an information transmission channel and assume directions distort independently we can use the water-filling algorithm to optimally allocate the bit width of each channel using $p_i$ as the price function, in that case the channel $i$ gets a bit width of $b_i = \text{clip}(\frac{1}{2} \log_2(p_i/\theta), b_{min}, b_{max})$ where $\theta$ is found by iterated bisections so mean bit width coincides with the budget. Experiments in modifying this price function are reported in findings. We can then assign a bit width to each column of the matrix $WV$ and quantize accordingly.

For the quantization itself we use 2 methods: 
- KLQ-RTN uses a simple RTN with $b_{min}=3$ and $b_{max} = 12$, 64-sized grouping which adds 0.25bpw to the final quantization numbers.
- KLQ-VQ uses additive vector quantization with 2 codebooks, $d=8$ and 256 levels, this applies only to directions quantized below 2.5 bits. Similarly we use 64-sized grouping and $b_{min}=2$.

These are deliberately simple choices as the quantization acts as an interchangeable backend, the method only applies to direction priorization and bit-width allocation. While quantizing we also reestimate the eigenbases per block under the already quantized upstream while KL price remain fixed.

For the KV-cache and activations we use the same procedure. Activations are sampled from the input of the MLP and the attention (post and res spaces), the KV is sampled from each layer and head, using a different eigenbasis per head, with the first 4 tokens assumed as sink and kept at full precision. This repo provides inference-side simulated quantization hooks, no actual kernels for deployment.

-------------------------------
## Findings

First of all, the study of the geometry of the model confirms the massive activations, $\lambda_1 \gg \lambda_i$ for all $i>1$. The rest of the directions form a sheath around this massive activation. 

Most allocation-axis methods use variance as a measure of the importance of a direction. Analyzing variance and causal KL damage turns out the correlation between these two vary wildly, mainly depending on the space we are analyzing within the layer. Interestingly, this correlation is mostly the same, not only within spaces of different layers of the same model, but among different models.

The following table shows the average Spearman coefficient of $\lambda_i$ and $KL_i$ across all layers of a model.
|     Space               |      Qwen 2.5 0.5B           |              Llama 3.2 1B       |    
|-------------------------|------------------------------|---------------------------------|
|    QKV                  |       +0.458                 |             +0.460              |  
|   Residual stream       |       +0.871                 |             +0.833              |  
| Post-Attention residual |        +0.953                |             +0.947              |
|    Context Space        |        +0.769                |             +0.717              |
|     MLP-in              |        +0.887                |             +0.781              |
|    MLP-up (int)*        |         +0.964               |             +0.917              | 

<sub>*For int, computational tractability on a 3090 required measuring only the top ~20% of directions by eigenvalue, plus a random tail sample. Since KL–variance agreement is strongest at the top of the spectrum, the full-spectrum coefficient would likely be lower.

The following figures plot put each direction in each space of a specific layer of the Qwen 2.5 0.5B model and the Llama 3.2 1B model respectively:

![image](https://github.com/Mallacan-Coder/KLQ/blob/main/figures/lambda_kl_plot_qwen_10.png)
![image](https://github.com/Mallacan-Coder/KLQ/blob/main/figures/lambda_kl_plot_llama3_6.png)

This means that for certain spaces the variance-monotonicity of some directions implies KL-monotonicity of those same directions so ranking by variance would be an efficient heuristic for these spaces (res, post, int) while failing for the others (qkv, ctx). This would solve the problem of finding the most to least important directions, but KLQ not only finds important directions but we also use how important each direction is related to the rest to assign a bit-width via waterfilling. Turns out that even if the ranking was perfectly monotone with each other, waterfilling with variance would be worse than waterfilling with KL since both measures have very different distributions. KL is typically either more concentrated or more spread out than variance, depending on the space, the following images show the first 512/1024 directions of a few spaces of different layers from Qwen 2.5 0.5B and Llama 3.2 1B.

![image](https://github.com/Mallacan-Coder/KLQ/blob/main/figures/distribution_llama3_ctx_10.png)
![image](https://github.com/Mallacan-Coder/KLQ/blob/main/figures/distribution_llama3_qkv_5png.png)
![image](https://github.com/Mallacan-Coder/KLQ/blob/main/figures/distribution_qwen_post_6.png)


So, even if $\lambda_i>\lambda_j$ implied $KL_i>KL_j$ distributing bit-width using waterfilling on $\lambda_i$ would result in a sub-optimal usage of resources. We test this hypothesis by using $p_i = \lambda_i||Wv_i||^2$ as variance-pricing. The following table uses perplexity on the FineWeb dataset on 512-token sequences, same 64-grouping method used.

|     Method              |  Quantization  |      Qwen 2.5 0.5B           |              Llama 3.2 1B       |    
|-------------------------|----------------|------------------------------|---------------------------------|
|    FP16                 |     16/16/16   |        21.82                 |             16.94               |  
|   KLQ-RTN-KL            |      4/16/16   |        **26.34**             |             **18.87**               |
|    KLQ-VQ-KL            |      3.5/16/16 |        33.23                 |             24.31               |  
| KLQ-RTN-Variance        |      4/16/16   |        34.45                 |             21.55               |
| KLQ-VQ-Variance         |      3.5/16/16 |        42.58                 |             34.52               |   
|    Uniform-RTN          |      4/16/16   |        44.14                 |             627.77              |

KLQ-x-KL uses $p_i=KL_i ||Wv_i||^2$, KLQ-x-Variance uses $p_i=\lambda_i ||Wv_i||^2$ and Uniform quantized all directions to the same bit-width.

Several experiments with lower quantizations have found that the geometry of the model changes significantly as it is actively being quantized. Remeasuring res and post's $KL_i$ after the weights are quantized improves the results at very low quantizations. The following table measures perplexity of Llama 3.2 1B quantized at 3.2/3/16 in WikiText-2 at 2048-token long sequences, last one uses the first 4 tokens at FP16 as activation sinks, this amounts to a negligible bpw increase ($\sim+0.03\text{bpw}$) though it's worth mentioning for transparency. We also scale the floor of the quantization of activations by 4 sigmas.

|     Method                                                          | Llama 3.2 1B   |    
|---------------------------------------------------------------------|----------------|
|    KLQ-VQ-KL with stale $v_i$ and $KL_i$ from FP16                  |   $\gg400$    |
|    KLQ-VQ-KL with fresh $v_i$ and stale $KL_i$ from FP16            |     92.33      |
|    KLQ-VQ-Variance with fresh $v_i, \lambda_i$ |    1002       |
|    KLQ-VQ-KL with fresh $v_i$ and remeasured $KL_i$                   |     60.98  |
|    + first 4 as activation sink + extended scale floor                                      |     **37.13**  |
|     A3 alone on fp16 weights                                          | 22.72   |
| ReSpinQuant's 3/3/3                                                   |   49.90 |  

More experiments on low-bit quantizations reveal how KLQ exhibits superadditive PPL increases that don't follow the usual rule: $log(PPL_{A \land B}) = log(PPL_{A}) + log(PPL_{B}) - log(PPL_{Base})$:


|     Quantization                                                          | Llama 3.2 1B Wikitext-2 PPL |   Expected ppl |   
|---------------------------------------------------------------------|----------------|-------|
|            16/3/3                                        |   23.68    |  29  |
|       3/3/3        |    161.98 | 47 |
|    3/16/3  |    62.31   |   
|    16/3/16                  |     22.72  |
| 16/16/3 |  12.5    |

We hypothesize this might happen because each forward pass for remeasuring $KL_i$ under each different quantization uses merely 2 windows of 256 tokens, more calibration tokens might help clean out noise and lower this number though we currently lack the compute to run all the necessary tests on a reasonable scale of time. Using this same remeasurement technique for 4-bit quantization yields worse results (17.5ppl) at 2 windows and 256-token sequences than using the stale FP16 measurements, I hypothesize this happens because of the small sampling size, increasing to 4 windows and 512 token sequences improves it to 16.5ppl but that's still above the 4/16/16 result at 13.36ppl 

Overall KLQ proposes a new paradigm for 2 out of the main 4 axes of quantization: Allocation and Basis based on information theory and causally measured damage, while leaving room for composition on the Rounding and Codebook parts of the quantization with other methods such as QuIP# or NestQuant.

----------------
## Limitations, edge-cases and proposed experiments.

As of now this project has several limitations, from most important to least important:

1. <ins>Better codebooks:</ins> KV3 alone is benign (12.5) and A3/KV3 is sub-additive, but on W3-damaged weights KV damage becomes superadditive (W3/KV3 = 62 vs 26 expected and even KV4 doesn't fully recover). Fresh geometry and prices are already applied here so stale geometry shouldn't be a problem. The most promising attack is therefore a cleaner W3 (better codebook), which removes the interaction's cause, plus asymmetric K/V budgets (--k_avg/--v_avg implemented, mostly untested).
2. <ins>Better rounding:</ins> As I mentioned, this repo ships with only VQ and RTN methods for quantizing, other methods such as GPTQ/LDLQ might push the perplexity numbers lower though theoretically in the measured eigenbasis the activation Hessian is diagonal, so GPTQ's cross-dimension error compensation largely degenerates to RTN. Whether compensation in a different basis can still help KLQ-allocated bits is an open research question. 
3. <ins>Computational cost:</ins> Right now the only optimization made on computing $KL_i$ is using only the top 20% on the massive int space and taking a small sample off the 80% tail. Theoretically, it'd be possible to approximate $KL_i$ analytically for most spaces given only $KL_i$ for res though this goes entirely untested.
4. <ins>Scaling:</ins> Theoretically, bigger models have more redundant geometry so this might work in KLQ's benefit, though modern models are trained specifically to eliminate this redundant geometry so that might work against KLQ. Further testing with both bigger and more modern models is needed. One proposed experiment is to apply KLQ to the models introduced in "Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models" which provides two different models (both available on HuggingFace), one trained classically which exhibits the uneven space, and another one trained in an outlier-safe regime which exhibits a naturally even space, if the hypothesis made by KLQ are true then the outlier-safe model should benefit from KLQ much less than the classically trained model.
5. <ins>Real benchmarks:</ins> While perplexity is a good proxy for the degradation of a model under quantization a bundle of real benchmarks is still to be run.
6. <ins>Packed kernels for real quantization:</ins> This is not a limitation of the method itself but more so of this repo, currently we use "fake" quantization in which the numbers are rounded realistically but the memory and computational footprint of inference is the same as that of an FP16 model. This is not a big problem since KLQ for now is more a of a theoretical framework.
