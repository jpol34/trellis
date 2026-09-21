# Fine-tuning precedent for laya's `choice` head

Laya ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) on Hugging Face,
source at [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)) is a non-autoregressive
decision model: given a `state` (text/email/ticket/JSON) and a typed question (`choice`, `score`,
`bool`), it returns calibrated probabilities in a single forward pass. `pip install laya` exposes
inference only — there is no `model.finetune(...)` call in the package API. The authors' own
fine-tuning path is a notebook, not a library entry point:
[`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
That notebook does cover the `choice` head specifically, so this project adapts it directly rather
than falling back to a from-scratch supervised script.

## Training loop shape

Each training example is built from the
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets) dataset via
`build_training_item()`:

- **State**: the JSON-parsed workflow state (the document/context text).
- **Question**: `q["type"] == "choice"`, `q["instructions"]` (the question text), and
  `q["criteria"]` (a dict of option keys to option text).
- **Label**: `gold_q["probabilities"]`, a dict mapping option keys to target probabilities.
  `keys = list(crit.keys())`; `target = [gold_q["probabilities"].get(k, 0.0) for k in keys]`;
  the argmax of `target` is the correct label index.

Loss combines a supervised cross-entropy term with an RLCD/GRPO-style policy-gradient term, not
either alone:

```
r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)   # group-mean baseline (GRPO-style)
logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
loss_rl = -(adv * logp).mean()
loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM
```

- **Reward**: a strictly proper scoring rule (`proper_reward`, a spherical/ranked-probability-score
  blend) applied to the model's reported probability distribution — this is what "RLCD" refers to
  here: reward for reporting calibrated probabilities, not just picking the right label.
- **Exploration**: zero-mean Gaussian noise added to logits (`sigma` annealed 0.4 → 0.1 over
  training) to sample a group of `GROUP_SIZE=4` candidate outputs per example; the policy-gradient
  update uses a group-mean baseline, which is the "GRPO-style" part.
- **CE term**: a standard cross-entropy term against the same target distribution runs alongside
  the RL term at equal weight (`1.0`), so the objective is not RL-only — it's supervised label
  fitting plus a calibration-shaping reward.
- **Hyperparameters**: 4 epochs over ~30k typed-decision questions; micro-batch 8/GPU, grad
  accumulation 4, effective batch 64; encoder LR `2.5e-5`, head LR `1.0e-4`; AdamW
  (`weight_decay=0.01`); cosine LR schedule.

Base (zero-shot) checkpoints score near chance on typed-decisions (0.362 English, 0.352
multilingual). The notebook's fine-tuned run reaches 0.766, above the comparison baseline (Jev,
0.727), in roughly 4–5 hours on 2×T4.

## Token budget and option-count constraint

Sequences are split into an option-prompt budget (`head_max_len`) and a document/state budget
(`max_len - head_max_len`):

- English checkpoint: `max_len = 512`, `head_max_len = 192`.
- Multilingual checkpoint: `max_len = 1024`, `head_max_len = 256`.

`head_max_len` is shared across *all* options in a `choice` question — it does not scale per
option. The model card documents the failure mode directly: a 77-option question (Banking77)
allocates only `(256 - 16) // 77 ≈ 3–4` tokens per label, and accuracy collapses to 0.425 against
a 0.870 reference. Neither source states an explicit numeric "recommended option count," but the
same arithmetic gives a practical ceiling: after ~16 tokens of fixed overhead, a 256-token budget
leaves ~240 tokens for options, i.e. roughly 20 options at ~12 tokens/option before per-option
budget gets tight enough to matter.

This project's candidate sets are 5 options (real value + 3 distractors + "not mentioned"), which
is well under both the ~20-option rule-of-thumb ceiling and the raw per-option token math even on
the smaller 192-token English head budget (192 - 16) / 5 ≈ 35 tokens/option. The token-budget
constraint that bites at high cardinality (Banking77-style) does not apply here; no special
handling (hierarchical splitting, budget increase) is needed for a 5-option head.

## GPU shape: 2×T4 (notebook) vs. single RTX 4090 (this project)

The notebook trains via `torchrun --nproc_per_node=2` with PyTorch `DistributedDataParallel`
(`dist.init_process_group("nccl")`, one process per GPU, dataset sharded `all_items[rank::world_size]`,
gradients synchronized via `DDP.backward()`). This is data parallelism across two free Kaggle T4s
for throughput — it is not splitting the model or relying on cross-GPU memory pooling; each T4
independently holds a full model replica.

**Decision: adapt to a single GPU, not follow the notebook's 2-GPU count.** This project's RunPod
pattern (`laya_bench.settings.runpod_gpu_type_id` default `"NVIDIA GeForce RTX 4090"`,
`stack/Dockerfile`) is a single-GPU pod. Reasons:

- The 2×T4 split exists because Kaggle's free tier caps a single session at one T4; it is a
  platform constraint of the source notebook, not a training requirement. Nothing in the loss,
  reward, or batching logic depends on having two devices.
- A single RTX 4090 (24 GB, Ada Lovelace) has more VRAM and materially higher throughput than a
  single T4 (16 GB, Turing) individually, and comparable-to-more aggregate throughput than 2×T4
  for this model size (0.4B params) once DDP's cross-GPU sync overhead is removed.
- Reproducing the effective batch size of 64 on one GPU is a config change, not an architecture
  change: run a single process (no `dist.init_process_group`, no `DDP` wrapper), keep micro-batch
  8, and raise gradient accumulation from 4 to 8.
- Staying on a single GPU keeps this project on its existing RunPod single-pod pattern instead of
  introducing multi-GPU pod provisioning, NCCL configuration, and `torchrun` orchestration solely
  to replicate a Kaggle free-tier limitation.

## Applicability of RLCD/GRPO to this project's narrower task

The notebook's task is broad typed-decisions fine-tuning across arbitrary option cardinality and
question types (`choice`, `score`, `bool`), where calibrated *probabilities* matter for downstream
automation thresholds (e.g., confidence-gated routing) — not just top-1 accuracy. This project's
task is narrower: closed-set discrimination over fixed 5-option candidate sets, evaluated (per this
project's README) primarily as accuracy against Jev and GPT-5.1 baselines.

This project keeps the notebook's combined RLCD+CE objective rather than stripping it down to
plain cross-entropy: the CE term is already present in the combined loss at equal weight, so
adopting the full objective costs nothing over a CE-only fine-tune and additionally shapes
well-calibrated probability outputs, which is useful if benchmark comparisons ever look at
probability quality (e.g. calibration curves, not just argmax accuracy) rather than accuracy alone.
Since the 5-option task sits comfortably inside the token budget that causes RLCD/GRPO's
high-cardinality failure mode elsewhere, there is no cardinality-driven reason to deviate from the
notebook's approach for the `choice` head.

## Sources

- [convaiinnovations/laya — Hugging Face model card](https://huggingface.co/convaiinnovations/laya)
- [NandhaKishorM/laya — GitHub](https://github.com/NandhaKishorM/laya)
- [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
