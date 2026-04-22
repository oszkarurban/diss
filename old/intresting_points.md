It's not a dual approach specific to EAGLE3 — it's the verification strategy determined by the user's sampling parameters, not by the draft algorithm.

The branch at eagle_info.py:312 checks sampling_info.is_all_greedy:

Greedy verification (temperature=0 or greedy sampling): argmax(target) == draft_token — exact match, fully lossless. This is the path you'd always hit with temperature=0.

Stochastic verification (temperature>0): Uses tree_speculative_sampling_target_only with rejection sampling. This is also lossless in the speculative decoding sense — it produces the exact same distribution as sampling from the target model alone. The rejection sampling math guarantees this: accept with probability min(1, P_target(token) / P_draft(token)), and on rejection, resample from the corrected residual distribution. The output distribution is provably identical to just running the target model with that temperature/top-k/top-p.

So both paths are lossless. The distinction is just: when the user asks for greedy decoding, you can use a simpler exact-match check. When the user asks for stochastic sampling (temperature > 0), you need the full rejection sampling kernel to preserve the target distribution. EAGLE3 itself doesn't dictate which path — it's purely about how the user wants to sample.

For your benchmarks with temperature=0, you'll always be on the greedy path.

/rds/user/ou222/hpc-work/diss/SpecForge/benchmarks/benchmarker/base.py
base.py (lines 151-170)
It's hardcoded at base.py:159: temperature=0. So all your benchmarks use greedy decoding, which means you're always hitting the exact-match verify_tree_greedy_func path — never the stochastic rejection sampling kernel.

Good question. With --cuda-graph-max-bs 1:

capture_bs = [1] → 1 graph per (topk, steps) config
Capture is fast (seconds per config)
But: every request runs in a batch of 1, padded to bs=1. No batching benefit from CUDA graphs.
With --cuda-graph-max-bs N (e.g., 8 or 32):

capture_bs = [1, 2, 3, 4, 5, 6, 7, 8] (for N=8 with spec decoding) → 8 graphs per config
The system can batch multiple concurrent requests and replay the appropriate graph
More graphs × more configs = more capture time and memory
For 20 (topk, steps) configs × 8 batch sizes = 160 CUDA graphs
Memory: still manageable (~160 × 3 MB ≈ 500 MB), but capture time grows significantly
For your HPC benchmarks with --cuda-graph-max-bs 1, you're running 1 request at a time anyway, so bs=1 is fine. In a production setting with concurrent requests, you'd want a larger max_bs — but that multiplies the number of graphs to capture per config.

Now let me write the full implementation plan.


MY TEXT: "THE POLICY that adjusts the hyperparameters could be adjusted by which of the requests? imagine that 5 requests flow in which one shold we listen to. but i think its valid to say for now that we can focus only for bs=1 (i assume bs is the same as num requsts)"

Good clarifications. To answer your question: a "batch" is the set of requests being processed together in one forward pass. With --cuda-graph-max-bs 1, the batch always has exactly 1 request, so per-batch = per-request. Perfect for now.

Let me update the plan with all three decisions


warmup_steps : int
        Number of decode steps to collect signal statistics before the
        policy becomes active.  During warmup the starting config is used.
    """

    def __init__(
        self, server_args: "ServerArgs", warmup_steps: int = 10





initialized this way
    # Dynamic speculative decoding
    self.rolling_accept_rate: float = 1.0  # EMA, init optimistic
    self.spec_signal_log: List[Dict] = []  # per-step signal + config log




I need to insert the dynamic config selection at the top of draft(), before any work happens. The key insight: the config selection must happen before _draft_preprocess_decode because that method uses self.topk and self.speculative_num_steps for KV cache allocation.

But wait — for KV cache, we always want to allocate for the MAX config. Let me think about this more carefully...

Actually, _draft_preprocess_decode allocates num_seqs * self.topk * self.speculative_num_steps cache slots. If we've already selected a smaller config, we'd allocate fewer slots — which is fine because we don't need slots for the larger configs we're not using. But we'd need to handle the case where the config changes between steps and previously allocated slots might not be enough.

The plan says to always allocate for max config. But that wastes memory when using small configs. Since KV cache is freed after each verify step, I think it's better to allocate exactly what the chosen config needs. The allocation is fresh each step.

Let me insert the config selection.


I see the pattern clearly now. During replay:

Input buffers are copied from forward_batch into buffers
The graph replays (writing outputs into the same buffers)
Results are extracted
For the signal buffers, during capture the EagleDraftInput gets signal_draft_entropy and signal_hidden_norm pointing to pre-allocated tensors. During draft_forward(), the entropy computation writes into them. After replay, the values are in those buffers ready to be read.

The key: during capture (line 277), I need to pass the signal buffers into EagleDraftInput. During replay, no copy is needed for signals — they're output-only (written by the graph, read after replay).


Good. Now I need to make sure the signal buffers from the graph runner also get passed in during replay(). Looking at the replay code, the forward_batch.spec_info is used but the signal buffers need to point to the graph runner's pre-allocated buffers. Let me check what happens to forward_batch.spec_info during replay.

Actually, looking at the capture code more carefully: during capture, spec_info is created with the buffer references. When the graph is replayed, draft_forward() writes into spec_info.signal_draft_entropy which points to buffers.signal_draft_entropy. Since these are the SAME tensor objects, after replay the values are in buffers.signal_draft_entropy.

But during replay, the forward_batch.spec_info is the CALLER's spec_info, not the capture-time one. The signal buffers on the caller's spec_info might be None. However, draft_forward() accesses spec_info.signal_draft_entropy — which is the forward_batch's spec_info, not the captured one.

Wait, I need to re-examine. During capture:

A ForwardBatch is created with spec_info pointing to capture-time EagleDraftInput
draft_forward() reads/writes forward_batch.spec_info
The graph captures all tensor operations
During replay:

Input buffers are copied into the captured tensors
The graph replays, operating on the captured tensors
Outputs are in the captured tensors
The signal_draft_entropy tensor from capture time IS the buffer tensor. So during replay, the graph writes entropy into buffers.signal_draft_entropy. I can read it from there after replay.

So in _collect_signals(), after a CUDA graph replay, I should read from self.cuda_graph_runner.buffers.signal_draft_entropy. But I also need to handle the eager path where forward_batch.spec_info.signal_draft_entropy is used.

Let me update _collect_signals to read draft entropy from the graph runner's buffers OR from the spec_info:


Let me trace the entire entropy flow end-to-end and fix all the consistency issues at once. The problem is: the eager path needs pre-allocated signal buffers too, but spec_info.signal_draft_entropy defaults to None.

Read eagle_worker.py (lines 95-104)
I need to add eager-path signal buffer allocation. But I can't do it in __init__ yet because self.device and the max buffer size aren't known until after super().init(). Let me add a lazy allocation in draft() instea

Now eagle_info.py — add target_entropy and rolling_accept_rate computation in verify(). The verify method runs in eager mode so there are no CUDA graph constraints.

Read eagle_info.py (lines 328-352)
The target_probs are only computed in the stochastic branch (temperature > 0). In the greedy branch (temperature = 0), there are no target_probs — only argmax(logits). But the user's benchmarks use temperature=0 (greedy). So I should compute target_entropy from the raw logits in BOTH branches.

Let me add entropy computation right after the greedy/stochastic branch, using the raw logits_output.next_token_logits which is available in both cases.

Read eagle_info.py (lines 381-400)
I'll add target_entropy computation right before the simulation check (after both greedy and stochastic paths have completed), then add rolling_accept_rate update in the per-request loop.

Now I need to make the target_entropy accessible to the EAGLEWorker. Currently I stored it on self._dynamic_spec_target_entropy in the EagleVerifyInput instance. But the worker reads signals from the EagleVerifyOutput. Let me trace how the worker accesses this.

In eagle_worker.py:verify(), the worker calls spec_info.verify(batch, logits_output, ...) which returns EagleVerifyOutput. The target entropy is stored on self (the EagleVerifyInput instance). The worker has access to spec_info after the call.

Let me check the verify() method in eagle_worker.py:

The extend runner swap is already partially handled — _apply_dynamic_spec_config sets self.cuda_graph_runner_for_draft_extend. But I need to verify the forward_draft_extend_after_decode method actually uses it. Let me check:

Grep "def forward_draft_extend_after_decode" (in /rds/user/ou222/hpc-work/diss/sglang/python/sglang/srt/speculative/eagle_worker.py)
1 line of output
Read eagle_worker.py (lines 1202-1251)
Read eagle_worker.py (lines 1251-1280)
Good — forward_draft_extend_after_decode() already uses self.cuda_graph_runner_for_draft_extend which is swapped in _apply_dynamic_spec_config(). No additional changes needed here.

However, I notice the extend graph runner uses self.speculative_num_steps + 1 as num_tokens_per_req. Since we capture one extend runner per num_steps value, and we swap self.speculative_num_steps in _apply_dynamic_spec_config, the prepare_extend_after_decode(batch, self.speculative_num_steps) call at line 1231-1233 will correctly use the current config's num_steps. Good.


Now I see organize_draft_results returns parent_list, top_scores_index, draft_tokens. The top_scores_index contains indices of the top-scoring candidates. I need to extract the actual scores for path_score. Let me check what organize_draft_results provides.

Read eagle_utils.py (lines 20-44)
I can see that top_scores.values gives the actual scores of the selected candidates. The function currently discards these values (it only returns top_scores.indices). I can compute path_score from the score_list directly after organize_draft_results returns, using the same flattened score_list.

But organize_draft_results consumes score_list via torch.cat — the original tensors are still available in the caller. Actually no, draft_forward() returns the 3-tuple, not the raw score_list. And in draft(), only the 3-tuple is received.

The simplest approach: compute path_score from top_scores_index and the flattened score_list. But we don't have access to the flattened scores in draft().

Better approach: compute path_score inside draft_forward() before returning, and return it as a 4th value. But that changes the return signature, which would affect the CUDA graph runner too.

Simplest safe approach: compute path_score right before calling organize_draft_results in draft_forward(), from the already-computed score_list.

Read eagle_worker.py (lines 969-983)
I can compute path_score from score_list right before organize_draft_results. The score_list contains per-step cumulative probabilities for each candidate. The mean of the top scores gives a measure of path quality.

Now let me verify path_score computation won't break the CUDA graph path. The score_list comes from draft_forward() which is either replayed from CUDA graph or run eagerly. In the CUDA graph path, draft_forward() doesn't run directly — the graph runner's replay() returns (parent_list, top_scores_index, draft_tokens). The score_list is NOT returned from the graph.

This means _last_path_score can only be updated in the eager path, not the CUDA graph path. Let me check where path_score computation should go.

Read eagle_worker.py (lines 681-700)
I see the problem. In the CUDA graph path, draft_forward() is replayed inside the graph — we don't have access to score_list. The path_score computation I added is inside draft_forward(), which means:

Eager path: _last_path_score is updated correctly
CUDA graph path: draft_forward() runs inside the graph, but self._last_path_score is a Python float, not a tensor — it CANNOT be captured in a CUDA graph. This would break!
Wait, actually draft_forward() is called by the graph runner during capture, and the Python assignment self._last_path_score = ... involves .mean().item() which does a GPU→CPU sync. This would break CUDA graph capture.

I need to move the path_score computation OUTSIDE of draft_forward(). It should be computed in draft() after the CUDA graph replay, using the returned top_scores_index — but we don't have the raw scores there either.

The cleanest fix: for CUDA graphs, add a pre-allocated path_score buffer (like we did for entropy), and write the mean score there. But that's complex.

Simpler approach: path_score can be approximated from top1_prob since the path score is essentially the cumulative probability along the tree path. For now, let me move the computation out of draft_forward() and compute it in draft() only for the eager path, documenting that CUDA graph path doesn't update it.

Actually even simpler: the computation uses torch.cat(score_list, ...) and torch.topk(...) which are tensor ops that CAN be captured in a CUDA graph if we write the result into a pre-allocated buffer instead of using .item(). But .item() does a GPU-CPU sync which blocks capture.

Let me restructure: compute the topk in draft_forward() and write to a pre-allocated buffer. Then read .item() outside, in draft().


Now let me write a quick summary of all assumptions and their coverage:

Audit Summary — Final State
7 signals implemented:

Signal	Computed where	Works with CUDA graphs	Works with eager
draft_entropy	draft_forward() → pre-allocated buffer	Yes (graph output buffer)	Yes (lazy-allocated buffer)
top1_prob	_collect_signals() from extend-phase topk_p	Yes (reads post-graph output)	Yes
top1_minus_top2	_collect_signals() from extend-phase topk_p	Yes	Yes
hidden_norm	_collect_signals() from extend-phase hidden_states	Yes	Yes
target_entropy	eagle_info.py:verify() from raw target logits	N/A (verify is always eager)	Yes
entropy_gap	Derived: target_entropy - draft_entropy	Yes	Yes
rolling_accept_rate	eagle_info.py:verify() per-req EMA update	N/A (verify is always eager)	Yes
Assumptions documented in code:

topk_p shape in _collect_signals() uses previous round's topk — documented in shape reasoning block
rolling_accept_rate initializes to 1.0 (optimistic) — during warmup, policy returns starting config
Adaptive normalizer warmup = 10 steps — signals are collected but policy returns starting config
Batch size = 1 currently tested — documented in dynamic_spec.py module docstring
num_draft_tokens doesn't affect CUDA graphs — post-graph pruning only
Vanilla EAGLE3 isolation verified: All 8 guard locations use self._dynamic_spec_enabled or get_global_server_args().enable_dynamic_speculative_decodin



---HOW DID YOU SOLVE THIS? ---
Exit code 1
=== Module list ===
Currently Loaded Modulefiles:
 1) rhel8/slurm   2) singularity/current   3) rhel8/global   4) cuda/12.1  

=== nvcc version ===
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0

CondaError: Run 'conda init' before 'conda activate'
answer: bash -l -c 'source hpc/unload_prepare.sh 2>&1 && conda activate sglang-dev && python test_dynamic_adjustment.py' 2>&1



The error is clear: out_cache_loc.reshape(1, 1, 2) fails because out_cache_loc has size 3. This happens when we capture a graph for (topk=1, num_steps=2) — the out_cache_loc is allocated as num_seqs * topk * speculative_num_steps = 1 * 1 * 2 = 2 slots, but _draft_preprocess_decode was called with the previous config's values.

The root cause: in _init_cuda_graphs_dynamic, we swap self.topk and self.speculative_num_steps but the out_cache_loc buffer in the graph runner is sized based on whatever the runner reads from self during __init__. Let me trace the exact issue.

Read eagle_draft_cuda_graph_runner.py (lines 110-119)
out_cache_loc is sized max_num_token * speculative_num_steps where max_num_token = max_bs * topk. So for (topk=1, steps=2): size = 1 * 1 * 2 = 2.

But in draft_forward() at line 912:


out_cache_loc = out_cache_loc.reshape(forward_batch.batch_size, self.topk, self.speculative_num_steps)
The issue is that self.topk and self.speculative_num_steps on the EAGLEWorker are being read by draft_forward() during capture. But the graph runner was created with topk=1, steps=2, so the buffer has size 2. Meanwhile draft_forward reads self.speculative_num_steps which should be 2, and self.topk which should be 1. So 1 * 1 * 2 = 2 matches size 2.

Wait, the error says shape '[1, 1, 2]' is invalid for input of size 3. Size 3 means the buffer was allocated for a DIFFERENT config. Let me check what happens in the iteration order:

Read eagle_worker.py (lines 336-375)
Now I see the problem. The starting config is (topk=1, steps=3), so the starting graph runner has out_cache_loc size = 1 * 1 * 3 = 3.

When we try to capture (topk=1, steps=2), we set self.topk=1, self.speculative_num_steps=2. The EAGLEDraftCudaGraphRunner.__init__ creates a new buffer of size 1 * 1 * 2 = 2. But draft_forward() is called during capture, and it does:


out_cache_loc = out_cache_loc.reshape(forward_batch.batch_size, self.topk, self.speculative_num_steps)
The out_cache_loc on the forward_batch comes from the graph runner's buffers — size 2. And self.topk=1, self.speculative_num_steps=2, so reshape to (1, 1, 2) on size 2 should work...

Unless the forward_batch.out_cache_loc is somehow still pointing to the OLD buffer. Let me look at the capture code more carefully:

Read eagle_draft_cuda_graph_runner.py (lines 281-360)
I see it now! Line 315: out_cache_loc=out_cache_loc uses the buffer from this runner's __init__. The buffer IS correctly sized for this config.

But wait — draft_forward() at line 912 reshapes using self.topk and self.speculative_num_steps. These are on the EAGLEWorker (self.eagle_worker). During _init_cuda_graphs_dynamic, we DO set these before creating the runner. But the draft_forward is a method on self (the EAGLEWorker), so it reads self.topk and self.speculative_num_steps.

Let me re-check: error says shape '[1, 1, 2]' is invalid for input of size 3. So out_cache_loc has 3 elements but we tried to reshape to (1, 1, 2). The buffer has size 3 = 1 * 1 * 3 (the starting config), not size 2.

The problem: the forward_batch.out_cache_loc at line 315 uses buffers.out_cache_loc which is THIS runner's buffer. But buffers.out_cache_loc was allocated in __init__ with self.max_num_token * self.speculative_num_steps. Let me check: for (topk=1, steps=2), max_num_token = 1*1 = 1, and out_cache_loc = 1*2 = 2.

But capture_one_batch_size slices the buffer at line 252:

Read eagle_draft_cuda_graph_runner.py (lines 246-275)
Line 246: out_cache_loc = buffers.out_cache_loc[: num_tokens * self.speculative_num_steps]

Where num_tokens = num_seqs * self.num_tokens_per_bs and self.num_tokens_per_bs = self.topk.

For config (topk=1, steps=2): num_tokens = 1*1 = 1, slice is [:1*2] = [:2], so out_cache_loc has 2 elements.

Then draft_forward reshapes to (1, 1, 2) — that should work on size 2!

Unless... the issue is that self.speculative_num_steps on the graph runner (line 75) is not the same as self.speculative_num_steps on the eagle_worker. Let me check:

Read eagle_draft_cuda_graph_runner.py (lines 75-89)
Found it. Line 78-79: the graph runner reads speculative_num_steps and topk from model_runner.server_args, NOT from self.eagle_worker. The server_args is the global server config that has the ORIGINAL values (num_steps=3, topk=1). We swap eagle_worker.topk and eagle_worker.speculative_num_steps, but server_args.speculative_num_steps stays at 3.

So for config (topk=1, steps=2):

eagle_worker.speculative_num_steps = 2 (swapped)
But server_args.speculative_num_steps = 3 (unchanged)
Graph runner reads self.speculative_num_steps = 3 from server_args
out_cache_loc allocated as 1 * 1 * 3 = 3
But draft_forward() reads eagle_worker.speculative_num_steps = 2
Tries to reshape size 3 to (1, 1, 2) → crash
The fix: the graph runner should read topk and num_steps from the eagle_worker, not from server_args. But we can't change the graph runner class. Instead, we should temporarily set server_args.speculative_num_steps and server_args.speculative_eagle_topk during capture.

Edit eagle_worker.py
Edit failed
Reason: what do you mean by temporarily? i want this to work all the time
The word "temporarily" refers to the capture loop only — we set server_args to each config's values, capture the graph, then move to the next config, then restore the starting config at the end. The values are set correctly for each graph capture, and restored to the starting config when done. It's not a hack — it's the standard pattern used by the vanilla code too (the vanilla code only captures one config so it doesn't need to restore).

Let me apply the fix with clearer documentation:

The fix: EAGLEDraftCudaGraphRunner.__init__ reads speculative_num_steps and topk from server_args, not from the eagle_worker instance. We now set server_args.speculative_eagle_topk and server_args.speculative_num_steps before each graph capture, and restore them to the starting config after all captures are done

The graph capture succeeded (15 draft + 5 extend configs captured). The runtime error is in the logging code during verify — an accept_index value exceeds the draft_toks array length. This happens when the dynamic config changes num_draft_tokens but there's a mismatch in the tree topology.

Let me add a temporary diagnostic to see the actual values when it crashes:

Read eagle_info.py (lines 449-473)
The issue: acc_indices contains values that index into draft_toks but exceed its length. draft_toks = candidates_cpu[i] has length self.draft_token_num. The accept_index values come from the tree traversal kernel. The accept_index holds global indices into the flattened draft_token array across the batch, not per-request indices.

Wait — let me re-read. accept_index has shape (bs, spec_steps + 1). After the verify kernel, accept_index[i, j] is the index into the i-th request's draft tokens... but looking at the code more carefully:

Read eagle_info.py (lines 256-275)
accept_index shape is (bs, spec_steps + 1). The values from verify_tree_greedy_func should be indices into the reshaped candidates which has shape (bs, draft_token_num). So accept_index[i, j] should be in [0, draft_token_num).

But self.spec_steps comes from the EagleVerifyInput which was set in draft(). And self.draft_token_num also comes from there. If these are consistent, the values should be in range.

The key question: is retrive_index.shape[0] (= bs) correct? What if some requests finished and the batch changed size between draft and verify?

Actually, I bet the issue is simpler. Let me check — accept_index is (bs, spec_steps + 1) — the indices are GLOBAL flattened indices, not per-request indices. Let me look at how they're used:

Read eagle_info.py (lines 407-431)
I see! Line 421: id = predict_cpu[idx] uses idx as an index into predict_cpu (NOT into candidates_cpu[i]). predict has shape per line 260-262: predict_shape = list(logits_output.next_token_logits.shape)[:-1] + [+1] — so it's (bs * draft_token_num + 1,) or (bs, draft_token_num + 1).

And accept_index[i, j] is a GLOBAL index into the flattened predict/draft_token array. For request i, the valid indices are [i * draft_token_num, (i+1) * draft_token_num).

But then in line 457: [draft_toks[idx] for idx in acc_indices] uses the SAME global index idx to index into draft_toks = candidates_cpu[i] which only has draft_token_num entries (the i-th request's candidates). A global index like 1 * draft_token_num + 2 would be out of range for draft_toks.

This is a pre-existing bug in the instrumentation code (from changes_logging_1), not from our dynamic spec changes. It happens when bs > 1 because acc_indices contains global indices. With vanilla --cuda-graph-max-bs 1, bs is always 1 and the global index equals the per-request index, so it never manifests.

The fix: convert global indices to per-request indices.


You're right — padding with zeros is technically correct but semantically useless. The model decided to increase topk because it's confident and wants more branches, but then the extra branches are dead on arrival. That defeats the purpose.

The real solution: when topk changes, re-run the extend phase with the new topk to produce real probabilities for all branches. The extend phase calls fast_topk(probs, self.topk) on the draft model's output — if we run it with the new topk, we get genuine top-k probabilities and token indices.

Let me trace exactly what the extend phase does:

Read eagle_worker.py (lines 1151-1180)
Grep "def capture_for_decode" (in /rds/user/ou222/hpc-work/diss/sglang/python/sglang/srt/speculative/eagle_worker.py)
1 line of output
Read eagle_worker.py (lines 1324-1343)
Line 1327: fast_topk(probs, self.topk) — this uses the CURRENT self.topk. So if we change self.topk BEFORE the extend phase runs, the extend will naturally produce topk_p with the new topk dimension.

Now let me look at the call sequence in forward_batch_generation:

Read eagle_worker.py (lines 441-470)
The sequence within one forward_batch_generation() call:


1. draft(batch)              ← _apply_dynamic_spec_config runs HERE, changes self.topk
2. verify(batch, spec_info)
3. forward_draft_extend_after_decode(batch)  ← calls capture_for_decode(logits, draft_input)
                                                which does fast_topk(probs, self.topk)
So the extend at step 3 ALREADY uses the new self.topk to produce topk_p for the next round. The problem is only at step 1 of the CURRENT round — the topk_p inherited from the PREVIOUS round's extend was made with the old topk.

The clean fix: move the config change to happen AFTER the extend phase, not before draft. That way:


Round N:
  1. draft(batch)     — uses config from round N-1's decision (topk_p shape matches)
  2. verify(batch)    — collect signals (target_entropy, rolling_accept_rate)
  3. extend(batch)    — uses current self.topk to produce topk_p
  4. _apply_dynamic_spec_config  ← HERE: reads signals, decides new config,
                                    changes self.topk for the NEXT extend... but extend already ran!
No, that doesn't work either — the extend already ran with the old topk.

Actually the correct approach: apply the config change BETWEEN verify and extend:


Round N:
  1. draft(batch)     — uses topk from round N-1's config change (topk_p matches)
  2. verify(batch)    — target signals collected
  3. APPLY CONFIG      ← change self.topk here
  4. extend(batch)    — fast_topk(probs, NEW self.topk) → topk_p has new shape

Round N+1:
  1. draft(batch)     — topk_p has shape (bs, new_topk), matches self.topk ✓
This way:

Draft always sees topk_p that matches self.topk
Extend produces topk_p for the new topk
The config change takes effect on the very next round
ALL branches have real probabilities — no padding

OK. Now I have the full picture. Here's what I understand:

The extend CUDA graph for num_steps=N is captured with num_tokens_per_bs = N+1. This is the maximum number of tokens per request that the extend phase processes (verified_id has at most N+1 tokens: up to N accepted + 1 from the target model).

If the old tree had old_steps=5 and accepted 5 tokens, verified_id has 6 tokens per request. If we then switch to new_steps=2 and try to use the extend graph for num_steps=2 (max 3 tokens per request), the 6 tokens won't fit.

So changing num_steps between verify and extend is NOT safe if the accepted length exceeds the new num_steps.

This means the only place to safely change ALL three parameters is: at the start of the round, before draft. And the topk_p shape mismatch must be handled.

But my earlier zero-padding approach was rejected as semantically wrong. And we can't re-run the extend because we don't have the logits anymore.

There's actually a third option I didn't consider: re-run fast_topk with the new topk on the cached logits. But we don't cache the logits either — capture_for_decode only saves topk_p, topk_index, and hidden_states.

So the real question is: can we correctly produce (bs, new_topk) from (bs, old_topk) topk_p/topk_index without the full probability distribution?

If new_topk < old_topk: yes, just truncate. We have the top-old_topk values and the top-new_topk are the first new_topk columns.

If new_topk > old_topk: no, not exactly. We only have the top-old_topk values. The old_topk+1 through new_topk-th values are unknown.

But: there's one more option. We could cache the logits from the extend phase specifically for this purpose. The capture_for_decode function has access to logits_output.next_token_logits — we just need to save them.

Read eagle_worker.py (lines 1324-1333)
Here's my assessment of the situation:

The core constraint: topk_p.shape[1] at the start of draft() must match self.topk used by select_top_k_tokens(). The topk_p is produced by capture_for_decode() in the extend phase using fast_topk(probs, self.topk).

Three approaches to change topk dynamically:

Cache the full probs tensor from extend, recompute fast_topk(probs, new_topk) at the start of the next draft. Extra memory (~vocab_size × bs × float16 per step) but semantically perfect — all branches have real probabilities.

Apply config between verify and extend — works for topk (extend produces new shape), but breaks for num_steps because accept_length from the old tree may exceed the new extend graph's capacity.

Apply config at start of draft + adapt topk_p — truncation is correct for reducing topk, but increasing topk requires data we don't have.

Approach 1 is the cleanest but caches a large tensor. Approach 2 works for topk but not num_steps. Approach 3 has the "zero-padding is useless" problem.

My recommendation: Approach 2 (split the config change).

topk: change between verify and extend (extend produces real probabilities)
num_steps + ndt: change at the start of the next round's draft (these don't affect topk_p shape, only tree depth and pruning)
The policy decides all three params together, but they're applied at different points:

End of round N: set self.topk → extend produces correct topk_p
Start of round N+1: set self.speculative_num_steps and self.speculative_num_draft_tokens → draft builds tree with new depth
This means the topk change takes effect one step EARLIER than num_steps/ndt, but since the policy decides them together based on the same signals, the result is still coherent. Does this approach make sense to you?

intresting_points.md
maybe illustrate it i cant seem to understand
Let me draw the exact data flow for one complete cycle, showing where each tensor is created and consumed:


══════════════════════════════════════════════════════════════
 ROUND N  (current config: topk=1, steps=3, ndt=4)
══════════════════════════════════════════════════════════════

 batch.spec_info.topk_p = [[0.85]]          ← shape (1, 1) from Round N-1's extend
 batch.spec_info.topk_index = [[4523]]       ← shape (1, 1)
 batch.spec_info.hidden_states = [[...]]     ← shape (1, 4096)

 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 1: DRAFT                                         │
 │                                                         │
 │  select_top_k_tokens(step=0, topk_p, topk_index,       │
 │                      hidden_states, scores, topk=1):    │
 │                                                         │
 │    input_ids = topk_index.flatten()                     │
 │              = [4523]                    ← 1 token      │
 │                                                         │
 │    hidden_states = hs.repeat_interleave(topk=1, dim=0)  │
 │                  = [[...]]              ← 1 × 4096      │
 │                                                         │
 │    ✅ MATCH: 1 input_id, 1 hidden_state                 │
 │                                                         │
 │  ... draft model forward × (steps-1) times ...          │
 │  ... organize_draft_results(ndt=4) ...                  │
 │  ... build_tree(topk=1, steps=3, ndt=4) ...             │
 └─────────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 2: VERIFY                                        │
 │                                                         │
 │  target model evaluates 4 draft tokens                  │
 │  accepts 3 tokens (accept_length=3)                     │
 │  verified_id = [4523, 291, 8832]                        │
 │                                                         │
 │  → signals collected: target_entropy, rolling_accept_rate│
 └─────────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 3: EXTEND                                        │
 │                                                         │
 │  draft model processes verified_id (3 accepted tokens)  │
 │  produces logits for the LAST token position            │
 │                                                         │
 │  capture_for_decode:                                    │
 │    probs = softmax(logits)        ← shape (1, vocab)    │
 │    topk_p = fast_topk(probs, self.topk=1)               │
 │           = [[0.72]]              ← shape (1, 1)  ✅    │
 │    topk_index = [[9917]]          ← shape (1, 1)        │
 │    hidden_states = [[...]]        ← shape (1, 4096)     │
 │                                                         │
 │  These are stored in batch.spec_info for Round N+1      │
 └─────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════
 ROUND N+1  — POLICY SAYS: change topk from 1 → 3
══════════════════════════════════════════════════════════════

 batch.spec_info.topk_p = [[0.72]]          ← shape (1, 1) !!! from Round N extend
 batch.spec_info.topk_index = [[9917]]       ← shape (1, 1)

 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 1: DRAFT  (self.topk is now 3)                   │
 │                                                         │
 │  select_top_k_tokens(step=0, topk_p, topk_index,       │
 │                      hidden_states, scores, topk=3):    │
 │                                                         │
 │    input_ids = topk_index.flatten()                     │
 │              = [9917]                    ← 1 token      │
 │                                           (from 1 col)  │
 │                                                         │
 │    hidden_states = hs.repeat_interleave(topk=3, dim=0)  │
 │                  = [[...], [...], [...]] ← 3 × 4096     │
 │                                                         │
 │    ❌ MISMATCH: 1 input_id, 3 hidden_states             │
 │                                                         │
 │    Model forward gets 1 embed but 3 hidden_states       │
 │    torch.cat([embeds, hidden_states]) → CRASH           │
 └─────────────────────────────────────────────────────────┘
The problem: topk_p has 1 column (from the extend that ran with topk=1), but select_top_k_tokens expands hidden_states by the NEW topk=3. The input_ids comes from topk_index.flatten() which has bs × old_topk = 1 element, while hidden_states has bs × new_topk = 3 rows.

The fix (split config change):


══════════════════════════════════════════════════════════════
 ROUND N  (config: topk=1, steps=3, ndt=4)
══════════════════════════════════════════════════════════════

 PHASE 1: DRAFT  — topk=1, steps=3, ndt=4
 PHASE 2: VERIFY — collect signals

 ★ POLICY DECIDES: topk=3, steps=4, ndt=6
 ★ APPLY topk=3 NOW (before extend)
 ★ SAVE steps=4, ndt=6 FOR NEXT ROUND

 PHASE 3: EXTEND
   capture_for_decode:
     topk_p = fast_topk(probs, self.topk=3)    ← NEW topk!
            = [[0.72, 0.15, 0.08]]              ← shape (1, 3) ✅
     topk_index = [[9917, 302, 1847]]           ← shape (1, 3) ✅

══════════════════════════════════════════════════════════════
 ROUND N+1
══════════════════════════════════════════════════════════════

 ★ APPLY steps=4, ndt=6 NOW (before draft)
 
 batch.spec_info.topk_p = [[0.72, 0.15, 0.08]]  ← shape (1, 3) ✅

 PHASE 1: DRAFT  (topk=3, steps=4, ndt=6)
   select_top_k_tokens(step=0, ..., topk=3):
     input_ids = [9917, 302, 1847]        ← 3 tokens  ✅
     hidden_states = repeat_interleave(3)  ← 3 rows    ✅
     MATCH ✅

   ... all 3 branches have REAL probabilities ...
The key: topk is applied one step earlier (at extend) so the probabilities are real, while num_steps and ndt are applied at the next draft because they don't affect tensor shapes, only tree depth and pruning.

---

Confirmed: line 400 — the extend graph captures fast_topk(probs, self.topk) inside the CUDA graph. self.topk here is the extend graph runner's self.topk (set from server_args.speculative_eagle_topk at line 81).

Now I have the full picture. Here's the best approach:

Option F: Always extend with topk_max, truncate at draft start
Core idea: Make the extend phase ALWAYS produce topk_p with shape (bs, topk_max) — the maximum possible width. At the start of each round's draft, truncate to (bs, current_topk).

Why this works:

fast_topk(probs, topk_max) returns the top-topk_max probabilities sorted descending
Truncating topk_p[:, :k] gives exactly the top-k — mathematically identical to calling fast_topk(probs, k) directly
All branches have real probabilities, not zeros
Truncation is a tensor slice (nanoseconds, no GPU compute)
Why this is better than the alternatives:

Approach	Correctness	Latency overhead	Complexity
Zero-padding	Wrong (dead branches)	None	Low
Split application (topk at extend, steps at draft)	Correct	None	High (two application points)
Cache full probs	Correct	None runtime, 256KB memory	Medium
Option F: extend with topk_max + truncate	Correct	None	Low
Concrete scenario walkthrough:

Config: topk_max=4, steps_max=5. Starting: topk=1, steps=3, ndt=4.


Round N (config: topk=1, steps=3, ndt=4):
  draft: topk_p truncated from (bs,4) to (bs,1) → linear chain
  verify: accept 3 tokens
  extend: fast_topk(probs, topk_max=4) → topk_p = [[0.72, 0.15, 0.08, 0.03]]  shape (bs,4)

Round N+1 — policy says (topk=3, steps=5, ndt=8):
  _apply_dynamic_spec_config: self.topk=3, self.steps=5, self.ndt=8
  draft: topk_p truncated from (bs,4) to (bs,3) → [[0.72, 0.15, 0.08]]
         ALL 3 branches are REAL probabilities ✅
         select_top_k_tokens: input_ids=3 tokens, hidden_states=3 rows ✅
         Draft graph for (topk=3, steps=5) selected
         KV cache for 3×5=15 slots
  verify: accept up to 5 tokens
  extend: fast_topk(probs, topk_max=4) → shape (bs,4) again

Round N+2 — policy says (topk=1, steps=2, ndt=3):
  draft: topk_p truncated from (bs,4) to (bs,1) → [[0.85]]
         Top-1 from the full distribution ✅
         Draft graph for (topk=1, steps=2) selected
Losslessness: Unaffected. The topk/steps/ndt parameters only control which draft tokens are proposed. The verification (greedy argmax match or stochastic rejection sampling) is identical — it accepts only tokens that match the target model's distribution.

Latency: Zero overhead. fast_topk(probs, topk_max) vs fast_topk(probs, topk) — the difference is negligible (topk_max is small, e.g., 4). The truncation is a tensor slice.

Vanilla EAGLE3: When enable_dynamic_speculative_decoding=False, capture_for_decode uses self.topk as before. No change.

Changes needed:

capture_for_decode(): use self.topk_max instead of self.topk when dynamic spec enabled
Extend CUDA graph capture: ensure server_args.speculative_eagle_topk = topk_max so the graph's fast_topk uses topk_max
_apply_dynamic_spec_config(): after setting self.topk, truncate batch.spec_info.topk_p[:, :self.topk]
Config change happens at START of draft (all three params together, one place, simple)

The root cause is clear. The target model's CUDA graph runner has num_tokens_per_bs fixed at server startup from server_args.speculative_num_draft_tokens. When dynamic spec changes num_draft_tokens (e.g., from 4 to 6), the target model's verify graph tries to process 6 tokens per request but its buffers are sized for 4.

This is NOT about the draft graphs (those we handle correctly with the multi-config library). It's the target model that also has CUDA graphs for the verify phase, and those graphs assume a fixed draft_token_num.

The fix: the target model's verify graph needs to be captured with the MAX num_draft_tokens. Let me check exactly where this is set: