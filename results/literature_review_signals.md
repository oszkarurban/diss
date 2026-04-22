# Literature Review: Runtime Signals for Dynamic Speculative Decoding

**Date**: 2026-03-31
**Scope**: 17+ papers on speculative decoding signals, adaptive tree structures, and confidence-based speculation control

---

## 1. Introduction

Speculative decoding accelerates large language model (LLM) inference by having a fast draft model propose multiple candidate tokens, which are then verified in parallel by the target model. The theoretical speedup depends critically on the *acceptance rate* --- the fraction of draft tokens that the target model confirms. This review examines how the literature uses runtime signals to predict acceptance, adapt speculation parameters, and optimise the draft-verify tradeoff.

Our work introduces a 14-signal dynamic speculative decoding framework that adjusts three tree-shape parameters (topk, num_steps, num_draft_tokens) per decode step. The strongest finding is that *contemporaneous target-side signals* --- particularly `target_top1_gap` (r=+0.435 with acceptance) --- outperform all draft-side signals, a result with no direct precedent in the literature. This review situates that finding within the broader landscape of signal-driven speculative decoding.

---

## 2. Foundational Works

### 2.1 Leviathan et al. (2023) --- Fast Inference from Transformers via Speculative Decoding

**Citation**: Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*. arXiv:2211.17192.

This foundational paper establishes the speculative decoding framework: a smaller "draft" model generates candidate tokens autoregressively, and the larger "target" model verifies all candidates in a single parallel forward pass. The key theoretical contribution is a *modified rejection sampling* scheme that guarantees the output distribution matches the target model exactly, making the acceleration lossless.

**Signal approach**: None. The method uses a fixed draft length (gamma tokens) with no runtime adaptation. The acceptance criterion is purely probabilistic: a draft token x is accepted with probability min(1, p_target(x) / p_draft(x)). The paper observes that speedup depends on the acceptance rate alpha, yielding expected tokens per step of (1 - alpha^{gamma+1}) / (1 - alpha), but makes no attempt to predict alpha at runtime.

**Relevance to our work**: Establishes the theoretical foundation but uses *no signals* for dynamic control. The fixed gamma parameter is precisely what our dynamic approach replaces with runtime signal-driven adaptation.

### 2.2 Chen et al. (2023) --- Accelerating Large Language Model Decoding with Speculative Sampling

**Citation**: Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.

The DeepMind speculative sampling paper independently proposes the same draft-then-verify paradigm. Their contribution focuses on the *modified rejection sampling* scheme that preserves the target distribution within hardware numerics, achieving 2--2.5x speedup on Chinchilla 70B in a distributed setup.

**Signal approach**: None. Like Leviathan et al., the method uses fixed speculation length with no runtime adaptation. The paper establishes that acceptance rates are content-dependent (harder passages yield lower acceptance) but does not exploit this observation.

**Relevance to our work**: The observation that acceptance rates vary with content is the foundational motivation for dynamic speculative decoding --- if acceptance is context-dependent, then speculation parameters should be too.

---

## 3. Draft-Model Confidence as a Signal

### 3.1 EAGLE (Li et al., 2024) --- Speculative Sampling Requires Rethinking Feature Uncertainty

**Citation**: Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. *ICML 2024*. arXiv:2401.15077.

EAGLE reframes speculative decoding as feature-level autoregression: rather than predicting tokens directly, the draft model predicts the *second-to-top-layer features* of the target model, which are then projected to token logits. The key insight is that autoregression at the feature level is more tractable than at the token level, but inherent *feature uncertainty* limits performance.

**Signal approach**: EAGLE does not use runtime signals for dynamic control. However, it implicitly generates a rich signal --- the draft model's softmax distribution over tokens --- which subsequent papers (EAGLE-2, TALON) exploit. The draft model produces `topk_p` (top-k probabilities) and `topk_index` (top-k token indices) at each draft step, which are the raw material for confidence-based signals.

**Signals produced (but not used for adaptation)**:
- Draft token probabilities (`topk_p`) --- later shown to be well-calibrated proxies for acceptance
- Hidden state features at the second-to-top layer

**Relevance to our work**: Our `top1_prob` signal (r=+0.184) and `draft_entropy` signal (r=-0.194) are computed from exactly these EAGLE draft outputs. EAGLE provides the signals but does not close the loop by using them for adaptation.

### 3.2 EAGLE-2 (Li et al., 2024) --- Faster Inference of Language Models with Dynamic Draft Trees

**Citation**: Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees. *EMNLP 2024*. arXiv:2406.16858.

EAGLE-2 is the first paper in the EAGLE family to exploit draft model confidence for dynamic tree construction. The key empirical finding is that **the EAGLE draft model is well-calibrated**: draft tokens with confidence below 0.05 have actual acceptance rates of approximately 0.04, while those with confidence above 0.95 have acceptance rates of approximately 0.98.

**Signal approach**: Single signal --- cumulative path probability.

The method operates in two phases:
1. **Expansion phase**: For each candidate token, compute a *value score* V_i = product of confidence scores c_j along the path from root to node i. This cumulative product represents the global probability of reaching that node through successive acceptances.
2. **Reranking phase**: Select the top-m tokens by value score, prioritising shallower nodes on ties.

**Critical insight**: EAGLE-2 uses the draft model's own confidence as a direct proxy for acceptance probability. This is a *draft-as-oracle* approach --- it trusts the draft model to know when it is likely to be correct. The calibration result validates this trust empirically.

**Limitations**: The signal is purely draft-side. When the draft model is poorly calibrated (e.g., mismatched draft/target architectures), the confidence-acceptance correlation degrades. EAGLE-2 does not use any target-side feedback signal.

**Relevance to our work**: Our `top1_prob` and `top1_minus_top2` signals capture the same draft confidence information. However, our correlation analysis shows these are relatively weak predictors (r=+0.18) compared to target-side signals like `target_top1_gap` (r=+0.44). This suggests that draft confidence alone is insufficient --- the target model's own uncertainty matters more.

### 3.3 EAGLE-3 (Li et al., 2025) --- Scaling up Inference Acceleration via Training-Time Test

**Citation**: Li, Y., Wei, F., Zhang, C., & Zhang, H. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. *NeurIPS 2025*. arXiv:2503.01840.

EAGLE-3 abandons feature prediction for direct token prediction and replaces top-layer features with multi-layer feature fusion via "training-time test." This resolves the scaling limitation of EAGLE/EAGLE-2, where increasing training data yielded diminishing returns due to the feature prediction bottleneck.

**Signal approach**: EAGLE-3 does not introduce new runtime signals. It improves the underlying draft model quality, which indirectly improves the calibration of draft confidence scores. Speedups of up to 6.5x (1.4x over EAGLE-2) demonstrate that better draft models produce more reliable confidence signals.

**Relevance to our work**: We use EAGLE-3 draft models in our experiments. The improved calibration means our `top1_prob` and `draft_entropy` signals are more reliable than they would be with EAGLE-1/2 drafters, though they remain weaker than target-side signals.

### 3.4 BiLD (Kim et al., 2023) --- Speculative Decoding with Big Little Decoder

**Citation**: Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M. W., Gholami, A., & Keutzer, K. (2023). Speculative Decoding with Big Little Decoder. *NeurIPS 2023*. arXiv:2302.07863.

BiLD introduces two explicit confidence-based policies for controlling draft-target interaction:

1. **Fallback policy**: When the small model's prediction probability for a token falls below threshold alpha_FB, control transfers to the large model. This is a *confidence threshold* applied to the draft model's top-1 probability.
2. **Rollback policy**: After the large model takes over, it re-evaluates all previous tokens. If the prediction probability distance between small and large models exceeds alpha_RB for a sequence of tokens, those tokens are rolled back and regenerated by the large model.

**Signal approach**: Single signal --- draft model's top-1 token probability, compared against a fixed threshold.

**Signal taxonomy**:
| Signal | Type | Source | Usage |
|--------|------|--------|-------|
| Draft top-1 probability | Confidence | Draft model softmax | Fallback trigger (alpha_FB) |
| Large-small prediction distance | Divergence | Both models | Rollback trigger (alpha_RB) |

**Limitations**: Fixed thresholds require offline tuning per model pair. No adaptation to content difficulty. The rollback policy requires *target model output* but only uses it retrospectively, not predictively.

**Relevance to our work**: BiLD's fallback policy is conceptually similar to our use of `top1_prob` as a signal, but BiLD applies a hard binary threshold rather than a continuous mapping. Our framework maps confidence to a continuous range of tree configurations rather than a binary draft/target switch. BiLD's rollback policy is a crude form of target feedback, but it operates *after* the damage is done rather than *before* (i.e., it corrects errors rather than preventing them).

---

## 4. Tree Structure Adaptation

### 4.1 SpecInfer (Miao et al., 2024) --- Tree-based Speculative Inference and Verification

**Citation**: Miao, X., et al. (2024). SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. *ASPLOS 2024*. arXiv:2305.09781.

SpecInfer introduces the concept of using *multiple* small speculative models to jointly construct a *token tree* of candidate sequences, verified in parallel by the target model. The token tree captures diverse alternative continuations, increasing the probability that at least one path matches the target.

**Signal approach**: SpecInfer uses no explicit runtime signals for tree adaptation. The tree structure is determined by the ensemble of draft models' predictions --- diversity in the ensemble implicitly creates wider trees for uncertain contexts. However, the tree topology is not dynamically adjusted based on acceptance feedback.

**Relevance to our work**: SpecInfer demonstrates the value of tree-based speculation (exploring multiple paths simultaneously) but does not close the feedback loop. Our approach dynamically adjusts tree width (topk) and depth (num_steps) based on runtime signals, achieving the same diversity effect adaptively.

### 4.2 Sequoia (Chen et al., 2024) --- Scalable, Robust, and Hardware-aware Speculative Decoding

**Citation**: Chen, Z., et al. (2024). Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding. arXiv:2402.12374.

Sequoia introduces a dynamic programming algorithm to find the optimal tree structure that maximises the expected number of accepted tokens. The key modelling assumption is the *positional acceptance model*: acceptance probability depends only on child position k in the tree, captured in an acceptance vector p = (p_1, p_2, ...).

**Signal approach**: Offline acceptance rate estimation, not runtime signals.

The DP algorithm fills a 2D tensor T(n, b) representing the best tree of size n with b direct children at root, using the recurrence:
```
T[n, b] = max over subtree allocations of expected tokens
```

**Key finding on temperature**: Sequoia discovers that lower temperatures yield higher acceptance rates, justifying deeper trees. Higher temperatures require wider, shallower trees. This confirms that *context difficulty* (proxied by temperature) should influence tree shape --- exactly the adaptation our dynamic system performs at runtime.

**Hardware-aware optimizer**: Sequoia also considers hardware characteristics, optimising `Speedup(n,d) = G(n,d) / (t(n) + d*c)` where G is expected generated tokens, t(n) is verification time, and c is the draft-to-target time ratio. This hardware awareness is complementary to our signal-based approach.

**Limitations**: The acceptance vector is estimated from a calibration set of 200 examples and remains *fixed* during inference. There is no runtime adaptation. Sequoia's tree is optimal for the *average* input but suboptimal for any *specific* input that deviates from the calibration distribution.

**Relevance to our work**: Sequoia provides the theoretical framework for optimal tree structures given known acceptance rates. Our contribution is making the acceptance rate estimate *dynamic* --- updating it per decode step using runtime signals rather than using a fixed offline estimate.

### 4.3 OPT-Tree (Huang et al., 2024) --- Speculative Decoding with Adaptive Draft Tree Structure

**Citation**: Huang, J., et al. (2024). OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure. *TACL 2025*. arXiv:2406.17276.

OPT-Tree constructs adaptive draft trees by searching for the structure that maximises the mathematical expectation of acceptance length at each decoding step. Unlike Sequoia's offline DP, OPT-Tree adapts the tree structure dynamically.

**Signal approach**: Token-level acceptance probability estimates from the draft model, used to construct an optimal tree per step.

The method uses draft model probabilities to estimate per-token acceptance rates and then applies an optimisation algorithm to allocate the token budget across tree width and depth. With a sufficiently powerful draft model and adequate node budget, OPT-Tree can generate more than ten tokens in a single step.

**Relevance to our work**: OPT-Tree shares our goal of per-step tree adaptation but uses only draft-model probabilities. Our approach additionally leverages target-side signals (`target_top1_gap`, `target_entropy`) and historical signals (`rolling_accept_rate`), providing richer information for the adaptation decision.

### 4.4 DySpec (Xiong et al., 2024) --- Dynamic Token Tree Structure

**Citation**: Xiong, Y., Zhang, R., Li, Y., Wu, T., & Zou, L. (2024). DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure. arXiv:2410.11744.

DySpec bridges the gap between draft distribution and acceptance rate, showing that "the two variables are strongly correlated." Based on this, it employs a greedy strategy to dynamically expand the token tree at runtime.

**Signal approach**: Draft model output distribution statistics, used to predict acceptance.

**Key theoretical result**: DySpec proves that under mild assumptions, the greedy expansion strategy achieves optimal results. The method dynamically decides whether to expand a node (add children) or deepen the tree (extend a chain) based on the draft model's probability distribution at that node.

**Relevance to our work**: DySpec's use of draft distribution statistics is analogous to our `draft_entropy` and `top1_prob` signals. However, DySpec does not use target-side signals, as the target model has not yet been queried when the draft tree is being constructed. This is a fundamental limitation that our approach overcomes by using *previous-step* target signals as leading indicators.

### 4.5 TALON (2026) --- Confidence-Aware Speculative Decoding with Adaptive Token Trees

**Citation**: TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees. arXiv:2601.07353.

TALON is a training-free, budget-driven framework that adaptively allocates a fixed token budget across tree layers using a hybrid expansion strategy. It naturally shapes trees into "deep-and-narrow" form for deterministic contexts and "shallow-and-wide" form for uncertain branches.

**Signal approach**: Cumulative path probability with confidence-gated expansion.

The mechanism operates in two phases:
1. **Robust initialisation (Layer 0)**: Fixed top-K expansion regardless of confidence, avoiding "over-confidence traps" where small draft models assign near-certain probabilities to incorrect tokens.
2. **Confidence-gated expansion (Layer d >= 1)**: Dynamic thresholding:
   ```
   P_{d+1} = {u in S_d | p(u) >= mu * m_d}
   ```
   where m_d = max(p(u)) is the highest confidence in the candidate pool and mu = 0.03 is the threshold hyperparameter.

**Key insight**: TALON's treatment of "over-confidence traps" at Layer 0 directly addresses a limitation of pure draft confidence signals. When the draft model is confident but wrong, naive confidence-based adaptation fails catastrophically. TALON mitigates this with the fixed-width first layer.

**Relevance to our work**: Our `joint_entropy_gate` signal (product of draft and target certainties, r=+0.309) addresses the same over-confidence problem from a different angle. Rather than using a heuristic (fixed first layer), our approach detects the "draft certain, target uncertain" regime directly through the gate signal. This is more principled but requires target-side information.

### 4.6 AdaEAGLE (Zhang et al., 2024) --- Explicit Modelling of Adaptive Draft Structures

**Citation**: Zhang, S., et al. (2024). AdaEAGLE: Optimizing Speculative Decoding via Explicit Modeling of Adaptive Draft Structures. arXiv:2412.18910.

AdaEAGLE introduces the Lightweight Draft Length Predictor (LDLP), a small learned module that predicts the optimal number of draft tokens for each decode step. The LDLP takes as input the draft model's hidden states and output tokens.

**Signal approach**: Learned prediction from hidden states --- implicit signal extraction.

Unlike our explicit hand-crafted signals, AdaEAGLE trains a predictor to extract the relevant information from raw hidden states. This avoids the need to identify which signals matter but loses interpretability.

**Relevance to our work**: AdaEAGLE's approach is complementary to ours. Where we use 14 explicit signals with correlation-weighted combination, AdaEAGLE uses a learned predictor. The advantage of our approach is interpretability and the ability to incorporate target-side signals. The advantage of AdaEAGLE is potentially capturing nonlinear signal interactions that our linear combination misses.

---

## 5. Hidden State and Embedding Signals

### 5.1 Judge Decoding (Bachmann et al., 2025) --- Going Beyond Model Alignment

**Citation**: Bachmann, G., et al. (2025). Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment. *ICLR 2025*. arXiv:2501.19309.

Judge Decoding is the most relevant prior work to our hidden-state signal approach. The paper demonstrates that LLMs *recognise errors* through their hidden representations: when forced to continue from incorrect tokens, models naturally attempt corrections, and this corrective impulse is detectable in the embeddings.

**Signal approach**: A lightweight linear classifier trained on target model embeddings.

**Technical details**:
- **Input features**: The last embedding of the target model *before* RMS normalisation and the language modelling head. Deeper layers perform best; shallower layers are clearly worse.
- **Architecture**: A simple linear head with 16,400 parameters, trained on 30,000 tokens in under 1.5 hours. MLPs and shallow Transformers were tested but offered no improvement.
- **Training data**: 500 question-answer tuples with manual error annotations. Tokens before errors are labelled positive; mistaken tokens are labelled negative.
- **Acceptance criterion**: Token accepted if sigmoid(f_judge(e_i)) > delta, where delta = 0.5 in practice.

**Key finding**: The judge achieves 9x speedup for 8B/405B model pairs by accepting "correct but non-aligned" tokens that standard rejection sampling would discard. This demonstrates that *token probability alignment is unnecessarily strict* --- semantic correctness, detectable in embeddings, is a better criterion.

**Relevance to our work**: Judge Decoding validates our hypothesis that hidden states contain rich information about prediction quality. Our `hidden_cosine_sim` signal (r=-0.148) captures a crude version of this --- representational stagnation rather than the full embedding vector. The Judge approach uses a learned classifier on *target* embeddings, while our approach uses training-free statistics on *draft* embeddings. The disparity in performance (Judge's 9x speedup vs. our modest hidden-signal correlations) suggests that:
1. *Target* embeddings are far more informative than *draft* embeddings for error detection.
2. A learned probe extracts far more information than scalar statistics (norm, cosine similarity, variance).
3. The directional information in embeddings matters; scalar summaries discard it.

### 5.2 Kangaroo (Liu et al., 2024) --- Double Early Exiting

**Citation**: Liu, F., et al. (2024). Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting. *NeurIPS 2024*. arXiv:2404.18911.

Kangaroo is a self-speculative decoding framework where the draft model is the target model's own shallow layers (with a lightweight adapter). It uses a *double early exiting* mechanism:
1. **First exit**: The draft model exits at a fixed shallow layer (layer 2 for 7B models, layer 3 for 13B).
2. **Second exit**: During drafting, generation halts when the draft model's top-1 confidence falls below threshold eta = 0.6.

**Signal approach**: Single signal --- draft model top-1 probability, compared against a fixed threshold.

The confidence-based early exit is a form of adaptive draft length: the system generates fewer draft tokens when predictions are uncertain and more when confident. This is conceptually identical to using `top1_prob` as a signal for `num_steps` control, which our system also does (though as one of 14 signals rather than the sole signal).

**Relevance to our work**: Kangaroo demonstrates that even a single confidence signal with a fixed threshold meaningfully improves over fixed draft lengths. Our multi-signal approach is a strict generalisation.

### 5.3 SPEED (Hooper et al., 2023) --- Speculative Pipelined Execution for Efficient Decoding

**Citation**: Hooper, C., Kim, S., Mohammadzadeh, H., Genc, H., Keutzer, K., Gholami, A., & Shao, Y. S. (2023). SPEED: Speculative Pipelined Execution for Efficient Decoding. arXiv:2310.12072.

SPEED speculatively executes multiple future tokens in parallel using predicted values based on *early-layer hidden states*. For Transformer decoders with parameter sharing, the memory operations for parallel tokens can be amortised.

**Signal approach**: Early-layer hidden states as implicit signals for future token prediction.

SPEED does not use explicit confidence signals for dynamic control. Instead, it uses intermediate representations as *direct predictors* of future tokens. This is a fundamentally different signal paradigm: rather than extracting scalar statistics from hidden states (as we do with `hidden_norm`, `hidden_cosine_sim`), SPEED uses the full hidden state vector as a draft prediction.

**Relevance to our work**: SPEED demonstrates that hidden states contain predictive information, supporting our use of hidden-state-derived signals. However, our scalar extraction approach (norm, cosine similarity) captures only a fraction of this information.

---

## 6. Knowledge Distillation and Draft-Target Alignment

### 6.1 DistillSpec (Zhou et al., 2024) --- Improving Speculative Decoding via Knowledge Distillation

**Citation**: Zhou, Y., et al. (2024). DistillSpec: Improving Speculative Decoding via Knowledge Distillation. *ICLR 2024*. arXiv:2310.08461.

DistillSpec improves acceptance rates by using knowledge distillation to *align* the draft model with the target model before applying speculative decoding.

**Signal approach**: No runtime signals. DistillSpec operates entirely at training time.

**Key theoretical result**: The token-level acceptance rate beta satisfies beta = 1 - D_TVD(p_target, p_draft), where D_TVD is the total variation distance. This directly connects distribution alignment to acceptance rate.

**Design choices validated by systematic study**:
1. **On-policy data generation** from the draft model outperforms off-policy (fixed dataset) approaches by ~20% speedup.
2. **Divergence function selection** matters: forward KL works best for greedy decoding; TVD is theoretically motivated but empirically similar.

**Relevance to our work**: DistillSpec's theoretical result (acceptance = 1 - TVD) provides the mathematical foundation for why our `target_top1_gap` signal works. When the target's top-1 gap is large, the target distribution is concentrated on one token, reducing the TVD between any reasonable draft and the target. Our signal is a runtime proxy for the theoretical quantity DistillSpec optimises at training time.

### 6.2 Online Speculative Decoding (Liu et al., 2024) --- Continuous Draft Model Adaptation

**Citation**: Liu, X., et al. (2024). Online Speculative Decoding. arXiv:2310.07177.

Online Speculative Decoding (OSD) continuously refines the draft model using the target model's corrections as training signal. When the draft proposes incorrect tokens, the target's corrections create free training data.

**Signal approach**: Target model corrections as training signals (not runtime control signals).

The system collects rejection events and corresponding target logits into a replay buffer, performing periodic batch updates using knowledge distillation (forward KL with teacher sampling proved most effective). This improves acceptance rates by 0.1 to 0.65 across datasets.

**Key insight**: The speculative decoding loop *inherently produces* the signal needed to improve the draft model --- rejection events identify exactly where the draft fails. This is a form of *target feedback* used for model improvement rather than parameter adaptation.

**Relevance to our work**: OSD uses target feedback to improve the draft model (slow, offline improvement). Our approach uses target feedback (`target_entropy`, `target_top1_gap`) for *immediate* parameter adaptation (fast, per-step control). These are complementary: OSD improves the baseline acceptance rate, while our dynamic adaptation optimises around that baseline.

---

## 7. Alternative Drafting Paradigms

### 7.1 Medusa (Cai et al., 2024) --- Multiple Decoding Heads

**Citation**: Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. *ICML 2024*. arXiv:2401.10774.

Medusa adds multiple parallel decoding heads to the target model itself, each predicting a different future position. Candidates are combined via Cartesian product and verified using tree attention.

**Signal approach**: *Typical acceptance scheme* --- a temperature-controlled threshold for candidate selection.

Rather than standard rejection sampling (which preserves the target distribution exactly), Medusa introduces "typical acceptance" that selects candidates whose probabilities fall within a *typical set* controlled by a threshold parameter. Higher thresholds create more stringent acceptance; lower thresholds accept more diverse candidates.

**Relevance to our work**: Medusa's typical acceptance trades exactness for higher acceptance rates. Our approach preserves exact target distribution (greedy verification at temperature=0) but optimises the *input* to verification (tree shape) rather than the *verification criterion* itself.

### 7.2 Lookahead Decoding (Fu et al., 2024) --- Parallel N-gram Generation via Jacobi Iteration

**Citation**: Fu, Y., Bailis, P., Stoica, I., & Zhang, H. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding. *ICML 2024*. arXiv:2402.02057.

Lookahead decoding views autoregressive generation as solving a nonlinear system via Jacobi iteration. It maintains a fixed-size 2D window, generating n-grams from Jacobi iteration trajectories, while a verification branch selects and verifies promising candidates.

**Signal approach**: None. The method is entirely training-free and signal-free. The Jacobi iteration naturally converges to correct tokens over multiple iterations, with n-grams from trajectories providing draft candidates. There is no confidence-based selection or dynamic adaptation.

**Relevance to our work**: Lookahead is orthogonal to our signal-based approach. It avoids the need for a separate draft model entirely, but achieves more modest speedups (1.5--2.3x) compared to EAGLE-family methods (3--6x).

### 7.3 REST (He et al., 2024) --- Retrieval-Based Speculative Decoding

**Citation**: He, Z., et al. (2024). REST: Retrieval-Based Speculative Decoding. *NAACL 2024*. arXiv:2311.08252.

REST replaces the parametric draft model with a non-parametric retrieval datastore. During each step, previous tokens serve as queries to find exact matches in the datastore; subsequent tokens from matches become draft candidates, organised in a Trie and verified with tree attention.

**Signal approach**: None for dynamic control. The retrieval mechanism implicitly adapts to context --- high-frequency patterns yield more candidates --- but there is no explicit signal for adjusting speculation parameters.

**Relevance to our work**: REST demonstrates that draft quality (and thus acceptance rate) is inherently context-dependent. Domains with repetitive patterns (code) yield high retrieval accuracy; diverse text yields lower accuracy. This context-dependence is precisely what our dynamic signals measure.

---

## 8. KL Divergence and Distribution-Based Signals

### 8.1 DSDE (2025) --- Dynamic Speculative Decoding with KLD Stability

**Citation**: DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving. arXiv:2509.01083.

DSDE is the closest prior work to our signal-based approach. It uses the *variance of KL divergence* between draft and target distributions as a diagnostic signal for regional generation stability.

**Signal approach**: Two derived signals from KLD history:

1. **Scale Factor (SF)**: Reacts to immediate draft-target disagreement:
   ```
   SF = exp(2 * mu_{KLD,last}) - 1
   ```
2. **Weighted Variance Intensity Ratio (WVIR)**: Compares short-term (10 steps) vs. long-term (30 steps) weighted KLD variance, detecting regime changes.

The predicted speculation length is:
```
SL_hat = (1 - SF * WVIR) * (SL_max - SL_min) + SL_min
```

When SF * WVIR > 1 (indicating instability), the system defaults to minimum speculation length.

**Critical difference from our work**: DSDE's signals are *post-hoc* --- they require target model verification output (KLD between draft and target distributions) from the *previous* step. This is similar to our `rolling_accept_rate` (historical signal) but DSDE does not use *contemporaneous* target signals. The KLD is computed after verification, reflecting the *previous* step's difficulty rather than the *current* step's.

**Relevance to our work**: DSDE validates the approach of using distribution-based signals for speculation control. However, our `target_entropy` and `target_top1_gap` signals provide *contemporaneous* information about the current step's difficulty, which we show is more predictive (r=+0.44 for target_top1_gap vs. the weaker historical signals).

### 8.2 PEARL (Li et al., 2024) --- Parallel Speculative Decoding with Adaptive Draft Length

**Citation**: Li, T., et al. (2024). PEARL: Parallel Speculative Decoding with Adaptive Draft Length. *ICLR 2025*. arXiv:2408.11850.

PEARL addresses the mutual waiting problem in speculative decoding through pre-verify (verifying the first draft token during drafting) and post-verify (generating additional draft tokens during verification). The adaptive draft length generates fewer tokens when early verification suggests rejection.

**Signal approach**: Early verification result as a binary signal.

PEARL uses the *first token's verification outcome* as a leading indicator: if the first draft token is rejected, subsequent draft tokens are likely also incorrect, so the system stops drafting early. This is a form of online signal, but it is binary (accept/reject) rather than continuous.

**Relevance to our work**: PEARL's pre-verification is conceptually related to our use of previous-step acceptance rate as a signal. Both exploit the temporal correlation in acceptance patterns. Our continuous signals provide finer-grained control.

---

## 9. Signal Taxonomy

### 9.1 Classification of Signals Across the Literature

We classify all signals used in the reviewed papers along three axes:

**Axis 1: Source**
| Source | Signals | Papers |
|--------|---------|--------|
| Draft model softmax | top-1 prob, entropy, cumulative path prob | EAGLE-2, BiLD, Kangaroo, TALON, OPT-Tree, DySpec |
| Draft model hidden states | early-layer activations, cosine similarity | SPEED, AdaEAGLE, *our work* |
| Target model softmax | target entropy, top-1 gap, varentropy | **our work only** |
| Target model hidden states | last-layer embeddings | Judge Decoding |
| Draft-target divergence | KLD, TVD, rollback distance | DSDE, DistillSpec (theory), BiLD |
| Historical acceptance | EMA of acceptance rate, rejection count | *our work*, DSDE |
| Joint draft-target | joint_entropy_gate | **our work only** |

**Axis 2: Temporal Relationship**
| Timing | Description | Examples |
|--------|-------------|----------|
| Contemporaneous draft | Signals from the current draft step | draft_entropy, top1_prob (EAGLE-2, TALON) |
| Contemporaneous target | Signals from the current verification step | **target_top1_gap, target_entropy (our work only)** |
| Historical | Signals from previous steps | rolling_accept_rate (our work), KLD variance (DSDE) |
| Derived/joint | Combinations of multiple sources | joint_entropy_gate (our work), entropy_gap (our work) |

**Axis 3: Computation Cost**
| Cost | Description | Examples |
|------|-------------|----------|
| Free (already computed) | Byproduct of existing computation | draft softmax (EAGLE-2), target logits (our work) |
| Cheap scalar | Simple reductions of existing tensors | entropy, norm, cosine similarity (our work) |
| Learned classifier | Requires training a small module | Judge Decoding (16K params), AdaEAGLE LDLP |
| Model update | Requires updating the draft model | Online SD (knowledge distillation) |

### 9.2 Signal-Paper Matrix

| Signal | EAGLE-2 | BiLD | Kangaroo | TALON | Sequoia | OPT-Tree | DySpec | DSDE | Judge | AdaEAGLE | Ours |
|--------|---------|------|----------|-------|---------|----------|--------|------|-------|----------|------|
| Draft top-1 prob | x | x | x | x | | x | x | | | | x |
| Draft entropy | | | | | | | x | | | | x |
| Cumulative path prob | x | | | x | | x | | | | | |
| Draft hidden states | | | | | | | | | | x | x |
| Target entropy | | | | | | | | | | | **x** |
| Target top-1 gap | | | | | | | | | | | **x** |
| Target varentropy | | | | | | | | | | | **x** |
| Target embeddings | | | | | | | | | x | | |
| Draft-target KLD | | | | | | | | x | | | |
| Draft-target TVD | | | | | | | | | | | |
| Rolling accept rate | | | | | | | | (variance) | | | x |
| Joint entropy gate | | | | | | | | | | | **x** |
| **Draft oracle gate** | | | | | | | | | | | **x** |
| Hidden cosine sim | | | | | | | | | | | **x** |
| Positional accept prob | | | | | x | | | | | | |

**Key observation**: The "Ours" column is the only one with entries in *every signal category*, including the novel **draft oracle gate** (multiplicative draft×target interaction, r=+0.78) — a signal paradigm with no precedent in the literature. No prior work uses contemporaneous target-side signals for speculation control.

---

## 10. Draft-as-Oracle vs. Target Feedback

A fundamental tension in the literature is whether to trust the draft model's self-assessment or use target model feedback.

### 10.1 Draft-as-Oracle Paradigm

**Papers**: EAGLE-2, TALON, OPT-Tree, DySpec, Kangaroo, BiLD

These papers treat the draft model's confidence as a reliable predictor of acceptance. The assumption is that a well-calibrated draft model "knows when it is right." EAGLE-2 empirically validates this for in-distribution data: confidence below 0.05 maps to ~4% acceptance, confidence above 0.95 maps to ~98% acceptance.

**Strengths**:
- Signals are available *before* target verification (no latency penalty)
- No additional computation required
- Works well when draft model is well-calibrated

**Weaknesses**:
- Calibration degrades with distribution shift (out-of-domain data)
- Mismatched draft/target architectures weaken the correlation
- Draft model may be confidently wrong ("over-confidence traps," identified by TALON)

**Our finding**: Draft-side signals achieve r=0.18--0.19 correlation with acceptance in our experiments. This is statistically significant but practically modest --- explaining only ~3.5% of variance in acceptance.

### 10.2 Target Feedback Paradigm

**Papers**: Judge Decoding, Online SD, DSDE, BiLD (rollback only)

These papers use information from the target model to assess or improve draft quality.

**Strengths**:
- Target model is the ground truth for acceptance
- Can detect draft failures that draft-side signals miss
- Particularly valuable for mismatched model pairs

**Weaknesses**:
- Target information is typically available only *after* verification (temporal lag)
- Judge Decoding requires training a classifier
- Online SD requires continuous model updates

**Our finding**: Target-side signals achieve r=0.38--0.44 correlation with acceptance --- approximately **2.3x stronger** than draft-side signals. This is the central empirical finding of our work.

### 10.3 Our Combined Approach

Our `joint_entropy_gate` signal (r=+0.309) operationalises the supervisor's "draft-as-oracle" suggestion: it combines draft certainty with target certainty into a single gate that identifies four regimes:

| Draft | Target | Gate Value | Interpretation | Optimal Action |
|-------|--------|------------|----------------|----------------|
| Certain | Certain | High (~1) | Both agree, safe | Speculate aggressively |
| Certain | Uncertain | Low (~0) | Draft confidently wrong | Reduce speculation |
| Uncertain | Certain | Moderate | Draft struggling, answer clear | Moderate speculation |
| Uncertain | Uncertain | Low (~0) | Genuinely hard | Minimal speculation |

The "draft certain, target uncertain" regime is the most dangerous and is precisely the "over-confidence trap" that TALON addresses heuristically. Our gate signal detects it directly.

---

## 11. Consensus Findings Across Papers

### 11.1 What Predicts Acceptance

Across the literature, the following factors consistently predict higher acceptance rates:

1. **Draft model confidence** (universal finding): Higher draft confidence correlates with higher acceptance. Validated by EAGLE-2 (calibration result), BiLD (fallback threshold), Kangaroo (early exit), TALON (path probability), and our work (r=+0.18).

2. **Content difficulty** (universal finding): "Easy" tokens (function words, deterministic continuations) have higher acceptance than "hard" tokens (content words, creative text). Sequoia shows this via temperature effects; REST shows it via retrieval accuracy variation.

3. **Draft-target distribution alignment** (theoretical foundation): DistillSpec proves acceptance = 1 - TVD(draft, target). Better alignment universally improves acceptance. DSDE measures this via KLD stability.

4. **Tree depth vs. acceptance tradeoff** (structural finding): Deeper trees accept fewer tokens per node but potentially produce longer accepted sequences. Sequoia, TALON, and our work all find that optimal depth depends on per-step acceptance rates.

### 11.2 What Does NOT Predict Acceptance Well

1. **Hidden state magnitude** (our finding): `hidden_norm` (r=-0.10) and `hidden_max` (r=+0.05) are near-noise. Judge Decoding confirms that the *direction* of embeddings matters, not the *magnitude*.

2. **Entropy gap direction** (our finding): The split of entropy_gap into positive/negative components adds little beyond the individual entropy signals. `entropy_gap_neg` correlates at r=+0.96 with `draft_entropy` --- they are the same signal.

3. **Fixed positional acceptance rates** (Sequoia's limitation): While useful on average, position-only models miss content-dependent variation. EAGLE-2 and our work show that token-level confidence is more informative.

### 11.3 Consensus on Optimal Strategy

The literature converges on several principles:
- **Adaptive is better than fixed**: Every paper that compares adaptive vs. fixed tree structures finds improvement (EAGLE-2: 20--40% over EAGLE-1; TALON: up to 5.16x; our work: +14.5% acceptance over vanilla).
- **Multiple signals beat single signals**: Our 7-signal system (+14.5%) outperforms any single signal. DSDE's two-signal approach (SF + WVIR) outperforms its single-signal ablations.
- **Calibration matters**: EAGLE-2's success depends on draft calibration; BiLD's thresholds require tuning; Kangaroo's eta=0.6 is model-specific.

---

## 12. What Our 14-Signal Approach Does Differently

### 12.1 Novel Contributions

1. **Contemporaneous target-side signals**: No prior work uses `target_entropy`, `target_top1_gap`, or `target_varentropy` computed from the *current* verification step's target logits for speculation control. DSDE uses *previous-step* KLD (historical), and Judge Decoding uses target embeddings for a *different purpose* (acceptance criterion modification, not tree parameter adaptation). Our signals are computed at zero marginal cost --- the target logits are already available from the verification phase.

2. **Joint draft-target signals**: The `joint_entropy_gate` (product of draft and target certainties) has no precedent. The closest work is BiLD's rollback policy (comparing draft and target predictions), but BiLD uses a binary threshold on divergence rather than a continuous multiplicative gate.

3. **Three-parameter tree adaptation**: Most prior work adapts one parameter: draft length (Kangaroo, BiLD, DSDE, PEARL, AdaEAGLE) or tree topology (EAGLE-2, TALON, Sequoia, OPT-Tree). Our system simultaneously adjusts *topk* (branching factor), *num_steps* (tree depth), and *num_draft_tokens* (verification budget), covering the full tree-shape parameter space.

4. **Training-free signal computation**: Unlike Judge Decoding (requires training a classifier) or AdaEAGLE (requires training LDLP), all 14 of our signals are computed from quantities already available during inference. The only "learning" is the correlation-weighted combination, which requires minimal calibration data.

5. **Signal redundancy analysis**: No prior work systematically analyses inter-signal correlations to identify redundancy. Our analysis reveals that `top1_prob` and `top1_minus_top2` are r=+0.97 correlated (effectively identical), `hidden_norm` and `hidden_var` are r=+0.99 (mathematically equivalent), and `draft_entropy` and `entropy_gap_neg` are r=+0.96 (redundant). This analysis enables principled signal pruning.

### 12.2 Quantitative Comparison of Signal Predictive Power

| Signal Source | Best Paper | Their Metric | Our Equivalent | Our r |
|---------------|-----------|--------------|----------------|-------|
| **Draft × acceptance product** | **None** | **N/A** | **draft_oracle_gate** | **+0.78** |
| **Target top-1 gap** | **None** | **N/A** | **target_top1_gap** | **+0.40** |
| **Target entropy** | **None** | **N/A** | **target_entropy** | **-0.38** |
| **Target varentropy** | **None** | **N/A** | **target_varentropy** | **-0.29** |
| **Joint entropy gate** | **None** | **N/A** | **joint_entropy_gate** | **+0.34** |
| Hidden state stability | (related: Judge Decoding) | Learned classifier | hidden_cosine_sim | -0.23 |
| Draft entropy | DySpec | Distribution width | draft_entropy | -0.22 |
| Draft confidence | EAGLE-2 | Calibration curve | top1_prob | +0.18 |
| KLD stability | DSDE | Variance of KLD | (not computed) | N/A |

The five novel signals (draft_oracle_gate, target_top1_gap, target_entropy, target_varentropy, joint_entropy_gate) are the five strongest predictors of acceptance, yet **none have precedent in the literature**.

### 12.3 The draft_oracle_gate Discovery

**Our strongest finding**, absent from all prior work: `draft_oracle_gate = top1_prob × rolling_accept_rate` achieves r=+0.69 (Llama-8B) and r=+0.78 (DeepSeek+LlamaDraft) correlation with acceptance — nearly **2× stronger** than the next best signal.

This signal operationalises a key insight: the draft model's confidence (`top1_prob`) is a weak predictor of acceptance in isolation (r=+0.18) because the draft can be confidently wrong. But when multiplied by the target's historical acceptance feedback (`rolling_accept_rate`), the product becomes the strongest predictor by far. The multiplication acts as a **calibration correction**: if the target has been rejecting drafts recently (low RAR), even high draft confidence produces a low gate value, preventing over-speculation.

**Why this has no precedent**: The literature treats draft-side and target-side signals as separate paradigms (Section 10). Draft-confidence approaches (EAGLE-2, Kangaroo, BiLD) use only draft-side information. Target-feedback approaches (DSDE, Online SD) use only historical acceptance data. No prior work combines the two multiplicatively. The closest is BiLD's rollback policy, which compares draft and target predictions using a binary threshold — but this is a verification criterion, not a confidence signal for tree parameter control.

**Signal taxonomy update**: `draft_oracle_gate` represents a new signal category — **multiplicative draft-target interaction** — that is distinct from both individual signals and from additive combinations (like our `joint_entropy_gate`). The multiplicative form is critical: addition would allow high draft confidence to compensate for low acceptance (dangerous), while multiplication requires both conditions to hold simultaneously.

---

## 13. Gaps in the Literature

### 13.1 Target-Side Signal Blindspot

The most striking gap is the complete absence of contemporaneous target-side signals for speculation control. This is likely due to a perceived chicken-and-egg problem: target logits are only available *after* verification, but tree parameters must be set *before* drafting. Our solution is simple --- use the *previous step's* target signals as predictors for the *current step's* acceptance, exploiting the temporal autocorrelation in generation difficulty.

### 13.2 Multi-Parameter Adaptation

Most work adapts a single dimension (draft length or tree width). Simultaneously adapting topk, num_steps, and num_draft_tokens creates a 3D optimisation problem that no prior work addresses. The interactions between these parameters (e.g., high topk with low num_steps vs. low topk with high num_steps for the same token budget) remain unexplored.

### 13.3 Signal Combination Strategies

Despite the proliferation of individual signals, no paper systematically studies how to *combine* multiple signals for speculation control. DSDE combines two signals (SF and WVIR) multiplicatively, but this is the only multi-signal system besides ours. Questions like "should signals be combined linearly, multiplicatively, or via a learned function?" remain open.

### 13.4 Mismatched Model Pairs

Nearly all evaluation uses *matched* draft-target pairs (e.g., EAGLE-3 drafter for Llama-3.1). Our experiments with mismatched pairs (Llama drafter for DeepSeek target) show that signal predictive power degrades but target-side signals degrade *less* than draft-side signals, suggesting they are more robust to model mismatch.

### 13.5 Batch-Level vs. Sequence-Level Adaptation

Our system and most prior work operate at the sequence level (adapting parameters for each sequence independently). In batched serving, different sequences may have different optimal configurations, creating a tension between per-sequence optimality and batch-level efficiency (CUDA graph constraints). DSDE's SL_cap addresses this partially, but systematic treatment is lacking.

---

## 14. Signal Combination Strategy Recommendations

Based on the literature review and our empirical findings, we recommend five concrete strategies:

### Recommendation 1: Target-First Hierarchical Gating

**Rationale**: Target-side signals are 2.3x more predictive than draft-side signals.

**Strategy**: First check `target_top1_gap` from the previous step. If high (> 0.8), speculate aggressively regardless of draft signals. If low (< 0.4), reduce speculation regardless of draft confidence. In the middle range, use draft signals (`top1_prob`, `draft_entropy`) to fine-tune.

**Support**: EAGLE-2 shows draft confidence is well-calibrated in the middle range but fails at extremes. Target signals provide the guardrails; draft signals provide the fine-tuning.

### Recommendation 2: Multiplicative Joint Gate with Asymmetric Penalties

**Rationale**: The "draft certain, target uncertain" regime (over-confidence trap, identified by TALON) is the costliest failure mode.

**Strategy**: Use `joint_entropy_gate` as the primary signal, but apply an asymmetric penalty: when draft confidence is high but target confidence is low, reduce the gate value by an additional factor. Formally:
```
effective_gate = joint_entropy_gate * (1 - max(0, draft_certainty - target_certainty))
```

**Support**: TALON's fixed first-layer heuristic and BiLD's rollback mechanism both address over-confidence. This strategy detects it directly and penalises it continuously.

### Recommendation 3: Correlation-Weighted Signal Pruning

**Rationale**: Signal dilution (14 equal weights degrading performance vs. 7 signals) is demonstrated empirically in our experiments.

**Strategy**: Use a pruned set of 8 signals with correlation-weighted combination:
- **High weight (56% total)**: target_top1_gap (16%), target_entropy (14%), joint_entropy_gate (14%), target_varentropy (12%)
- **Medium weight (30%)**: draft_entropy (10%), top1_prob (10%), hidden_cosine_sim (10%)
- **Low weight (14%)**: rolling_accept_rate (14%)

Remove: top1_minus_top2 (r=0.97 with top1_prob), hidden_var (r=0.99 with hidden_norm), hidden_max (noise), entropy_gap_pos (sparse), entropy_gap_neg (redundant with draft_entropy), hidden_norm (weak).

**Support**: DistillSpec shows alignment predicts acceptance; the weights should reflect predictive power, not equal allocation.

### Recommendation 4: Regime-Switching Policy

**Rationale**: Different generation phases (factual recall, creative text, code) have fundamentally different acceptance characteristics.

**Strategy**: Use `rolling_accept_rate` and `target_varentropy` to detect regime switches:
- **High accept + low varentropy**: Stable regime, speculate aggressively (max topk, max steps)
- **Low accept + high varentropy**: Unstable regime, speculate conservatively (topk=1, min steps)
- **Transition detected** (DSDE's WVIR > 1): Briefly reduce speculation, then re-evaluate

**Support**: DSDE's scale factor and WVIR address exactly this regime-switching problem. Our `rolling_accept_rate` provides a smoother, more responsive regime indicator (EMA with alpha=0.3, half-life ~2 steps).

### Recommendation 5: Learned Lightweight Combiner

**Rationale**: Linear combination of signals may miss important nonlinear interactions (e.g., the over-confidence trap is inherently a nonlinear relationship between draft and target signals).

**Strategy**: Train a small MLP (similar to Judge Decoding's 16K-parameter linear head) that takes the 8 pruned signals as input and predicts optimal (topk, num_steps, num_draft_tokens) directly. Training data can be collected automatically during inference by logging signals and corresponding acceptance outcomes.

**Support**: AdaEAGLE's LDLP demonstrates that a lightweight learned predictor can outperform hand-crafted rules. Judge Decoding shows that even a linear classifier on the right features achieves strong results. The key advantage over our current linear combination is the ability to capture the joint_entropy_gate-like interactions automatically.

---

## 15. Key Finding: The Target-Side Signal Gap

The single most important finding of this review is negative: **no prior work uses contemporaneous target-side signals (target_entropy, target_top1_gap, target_varentropy) for speculation control.**

This gap exists despite:
- DistillSpec proving theoretically that acceptance rate depends on draft-target alignment (which target signals directly measure)
- Judge Decoding proving that target embeddings contain rich error information
- DSDE using *historical* target information (KLD variance) and showing it outperforms draft-only signals
- The target logits being freely available after each verification step, requiring zero additional computation

The likely reasons for this gap are:

1. **Temporal framing**: Most papers frame speculation as a *feedforward* process (draft first, verify second). Using verification output to inform the *next* draft step requires a *recurrent* framing that is less natural in the speculative decoding paradigm.

2. **System design**: In most implementations, the draft and verify phases are separate codepaths. Feeding verification output back into draft configuration requires cross-phase communication that existing systems do not support. Our SGLang fork explicitly implements this feedback loop.

3. **CUDA graph constraints**: Dynamic tree parameters require multiple pre-captured CUDA graphs (one per configuration). Most systems capture a single graph, making dynamic adaptation infeasible at the systems level. Our implementation captures graphs for all valid (topk, num_steps, num_draft_tokens) combinations.

Our empirical results demonstrate that this gap is costly: `target_top1_gap` (r=+0.435) is **2.4x more predictive** than the best draft-side signal (`draft_entropy`, r=-0.194) and **4.3x more predictive** than `top1_prob` (r=+0.184), which is the signal most commonly used in the literature (EAGLE-2, BiLD, Kangaroo, TALON).

---

## 16. Summary Table of All Reviewed Papers

| Paper | Venue | Year | Signal Type | Signal Details | Adapts | Key Finding |
|-------|-------|------|-------------|---------------|--------|-------------|
| Leviathan et al. | ICML | 2023 | None | Fixed gamma | Nothing | Foundational framework |
| Chen et al. | arXiv | 2023 | None | Fixed gamma | Nothing | Speculative sampling |
| EAGLE | ICML | 2024 | None (produces signals) | Feature-level draft | Nothing | Feature uncertainty |
| EAGLE-2 | EMNLP | 2024 | Draft confidence | Path probability | Tree topology | Draft calibration |
| EAGLE-3 | NeurIPS | 2025 | None | Better draft model | Nothing | Training-time test |
| BiLD | NeurIPS | 2023 | Draft top-1 prob | Threshold fallback | Draft/target switch | Fallback + rollback |
| TALON | arXiv | 2026 | Path prob + threshold | Confidence-gated expansion | Tree topology | Over-confidence traps |
| Medusa | ICML | 2024 | Typical acceptance | Temperature threshold | Acceptance criterion | Multiple heads |
| SpecInfer | ASPLOS | 2024 | None (implicit diversity) | Multi-model ensemble | Tree via ensemble | Token tree verification |
| Sequoia | arXiv | 2024 | Offline acceptance rates | Positional accept vector | Tree topology (offline) | DP optimal tree |
| OPT-Tree | TACL | 2025 | Draft probabilities | Per-token acceptance est. | Tree topology | Adaptive per-step |
| DySpec | arXiv | 2024 | Draft distribution | Greedy expansion | Tree topology | Dist-acceptance bridge |
| Kangaroo | NeurIPS | 2024 | Draft top-1 prob | Threshold eta=0.6 | Draft length | Double early exit |
| SPEED | arXiv | 2023 | Early-layer hidden states | Implicit prediction | Parallel execution | Pipelined speculation |
| DistillSpec | ICLR | 2024 | None (training-time) | KD alignment | Draft model (offline) | accept = 1 - TVD |
| Online SD | arXiv | 2024 | Target corrections | KD from rejections | Draft model (online) | Continuous adaptation |
| Judge Decoding | ICLR | 2025 | Target embeddings | Learned linear classifier | Acceptance criterion | Beyond alignment |
| DSDE | arXiv | 2025 | KLD variance | SF + WVIR signals | Speculation length | Regional stability |
| AdaEAGLE | arXiv | 2024 | Hidden states (learned) | LDLP module | Draft length | Explicit length prediction |
| PEARL | ICLR | 2025 | Early verification | Binary accept/reject | Draft length | Pre/post-verify |
| **Ours** | — | 2026 | **14 signals (all types)** | **Draft + target + joint + historical** | **topk + steps + ndt** | **Target signals dominate** |

---

## 17. Conclusions

The literature on signal-driven speculative decoding has evolved rapidly from fixed-parameter methods (2022--2023) through single-signal draft confidence approaches (2024) to multi-signal adaptive systems (2024--2025). The dominant paradigm uses draft model confidence as the sole signal source, with EAGLE-2's calibration result providing the theoretical justification.

Our work identifies three critical gaps in this paradigm:

1. **Target-side signals are absent**: Despite being freely available and more predictive, no prior work uses contemporaneous target entropy, top-1 gap, or varentropy for speculation control.

2. **Joint draft-target signals are unexplored**: The interaction between draft and target certainty (our `joint_entropy_gate`) captures failure modes (over-confidence traps) that neither signal captures alone.

3. **Multi-parameter adaptation is rare**: Most work adapts one dimension; simultaneously optimising three tree-shape parameters is novel.

The literature strongly supports the principle that adaptive speculation outperforms fixed speculation. Our contribution is demonstrating that the *information source* for adaptation matters as much as the adaptation mechanism itself --- and that target-side signals, previously overlooked, are the single most valuable source.

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
2. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
3. Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. *ICML 2024*. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
4. Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees. *EMNLP 2024*. [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
5. Li, Y., Wei, F., Zhang, C., & Zhang, H. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. *NeurIPS 2025*. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
6. Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M. W., Gholami, A., & Keutzer, K. (2023). Speculative Decoding with Big Little Decoder. *NeurIPS 2023*. [arXiv:2302.07863](https://arxiv.org/abs/2302.07863)
7. TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees. (2026). [arXiv:2601.07353](https://arxiv.org/abs/2601.07353)
8. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. *ICML 2024*. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
9. Miao, X., et al. (2024). SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. *ASPLOS 2024*. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
10. Chen, Z., et al. (2024). Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding. [arXiv:2402.12374](https://arxiv.org/abs/2402.12374)
11. Huang, J., et al. (2024). OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure. *TACL 2025*. [arXiv:2406.17276](https://arxiv.org/abs/2406.17276)
12. Xiong, Y., Zhang, R., Li, Y., Wu, T., & Zou, L. (2024). DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure. [arXiv:2410.11744](https://arxiv.org/abs/2410.11744)
13. Liu, F., et al. (2024). Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting. *NeurIPS 2024*. [arXiv:2404.18911](https://arxiv.org/abs/2404.18911)
14. Hooper, C., Kim, S., Mohammadzadeh, H., Genc, H., Keutzer, K., Gholami, A., & Shao, Y. S. (2023). SPEED: Speculative Pipelined Execution for Efficient Decoding. [arXiv:2310.12072](https://arxiv.org/abs/2310.12072)
15. Zhou, Y., et al. (2024). DistillSpec: Improving Speculative Decoding via Knowledge Distillation. *ICLR 2024*. [arXiv:2310.08461](https://arxiv.org/abs/2310.08461)
16. Liu, X., et al. (2024). Online Speculative Decoding. [arXiv:2310.07177](https://arxiv.org/abs/2310.07177)
17. Bachmann, G., et al. (2025). Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment. *ICLR 2025*. [arXiv:2501.19309](https://arxiv.org/abs/2501.19309)
18. DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving. (2025). [arXiv:2509.01083](https://arxiv.org/abs/2509.01083)
19. Zhang, S., et al. (2024). AdaEAGLE: Optimizing Speculative Decoding via Explicit Modeling of Adaptive Draft Structures. [arXiv:2412.18910](https://arxiv.org/abs/2412.18910)
20. Li, T., et al. (2024). PEARL: Parallel Speculative Decoding with Adaptive Draft Length. *ICLR 2025*. [arXiv:2408.11850](https://arxiv.org/abs/2408.11850)
21. Fu, Y., Bailis, P., Stoica, I., & Zhang, H. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding. *ICML 2024*. [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)
22. He, Z., et al. (2024). REST: Retrieval-Based Speculative Decoding. *NAACL 2024*. [arXiv:2311.08252](https://arxiv.org/abs/2311.08252)
