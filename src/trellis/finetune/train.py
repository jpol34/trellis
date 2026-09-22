"""CLI (`trellis-finetune`) for fine-tuning laya's existing `choice` head on this project's
closed-set field-discrimination training corpus.

Follows the training-loop shape documented in `docs/finetune-precedent.md` (ticket #2): a
combined RLCD/GRPO policy-gradient term plus a supervised cross-entropy term against laya's
`DecisionModel` head, adapted from the notebook's 2xT4 DDP setup to a single GPU (micro-batch 8,
grad accumulation 8, for the same effective batch of 64). `laya` exposes inference only
(`laya.load()` / `Agent.predict()`); there is no `model.finetune(...)` entry point, so this
module drives `laya.common`'s lower-level building blocks (`build_sequence`, `collate_items`,
`proper_reward`) directly against the loaded `Agent`'s underlying `DecisionModel`.

Training examples are translated from the training-corpus JSONL schema into the real
`ChoiceQuestion` wire shape by `trellis.finetune.data` (ticket #4's adapter) before ever
reaching the model - never handed to the head as raw positional `candidates`/`correct_index`.

`--dry-run` runs one epoch over a tiny slice of the training data (a handful of steps, small
batch, small exploration group) so the training loop's wiring - forward pass, loss, backward
pass, checkpoint export - can be verified quickly on CPU without a real multi-hour GPU run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laya
import torch
from laya.common import QTYPES, build_sequence, collate_items, proper_reward
from safetensors.torch import save_file

from trellis.finetune.data import ChoiceExample, load_examples
from trellis.settings import settings

BASE_MODEL = "convaiinnovations/laya"
SEED = 700072

FULL_RUN_DEFAULTS: dict[str, Any] = {
    "epochs": 4,
    "batch_size": 8,
    "grad_accum": 8,
    "lr_encoder": 2.5e-5,
    "lr_head": 1.0e-4,
    "weight_decay": 0.01,
    "group_size": 4,
    "sigma_start": 0.4,
    "sigma_end": 0.1,
    "w_sph": 0.75,
    "w_rps": 1.0,
    "max_train_examples": None,
    "max_val_examples": None,
}

DRY_RUN_DEFAULTS: dict[str, Any] = {
    "epochs": 1,
    "batch_size": 2,
    "grad_accum": 1,
    "lr_encoder": 2.5e-5,
    "lr_head": 1.0e-4,
    "weight_decay": 0.01,
    "group_size": 2,
    "sigma_start": 0.2,
    "sigma_end": 0.2,
    "w_sph": 0.75,
    "w_rps": 1.0,
    "max_train_examples": 8,
    "max_val_examples": 4,
}


@dataclass
class TrainConfig:
    train_dir: Path
    val_dir: Path
    output_dir: Path
    base_model: str
    device: str | None
    seed: int
    epochs: int
    batch_size: int
    grad_accum: int
    lr_encoder: float
    lr_head: float
    weight_decay: float
    group_size: int
    sigma_start: float
    sigma_end: float
    w_sph: float
    w_rps: float
    dry_run: bool
    max_train_examples: int | None
    max_val_examples: int | None


def _build_item(tok, example: ChoiceExample, max_len: int, head_max_len: int) -> dict:
    q = {"t": "choice", "ins": example.instructions, "crit": example.criteria}
    ids, markers = build_sequence(tok, example.transcript, q, max_len, head_max_len)
    keys = list(example.criteria.keys())
    if len(markers) != len(keys):
        raise ValueError(
            f"item_id {example.item_id!r} field {example.field!r}: {len(keys)} options don't "
            f"fit head_max_len={head_max_len} (got {len(markers)} markers)"
        )
    target = [1.0 if k == example.correct_key else 0.0 for k in keys]
    label = keys.index(example.correct_key)
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": target,
        "label": label,
    }


def _make_batch(
    tok, examples: list[ChoiceExample], max_len: int, head_max_len: int, pad_id: int
) -> dict:
    items = [_build_item(tok, ex, max_len, head_max_len) for ex in examples]
    return collate_items([items], pad_id)


def _compute_loss(
    model,
    batch: dict,
    device: torch.device,
    sigma: float,
    group_size: int,
    w_sph: float,
    w_rps: float,
) -> torch.Tensor:
    """RLCD/GRPO policy-gradient term plus supervised cross-entropy against laya's `choice`
    head, per docs/finetune-precedent.md. `group_size` logit samples are drawn per example via
    zero-mean Gaussian noise (`sigma`); the policy-gradient update uses a group-mean baseline
    (the "GRPO-style" part). The trailing `+ 0.0 * act.sum()` keeps the unused `act` head in the
    autograd graph, matching the source notebook."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    marker_pos = batch["marker_pos"].to(device)
    mask = batch["marker_mask"].to(device)
    qtype = batch["qtype"].to(device)
    target = batch["target"].to(device)

    logits, act = model(input_ids, attention_mask, marker_pos, mask, qtype)

    noise = torch.randn((group_size, *logits.shape), device=device) * sigma
    z = (logits.unsqueeze(0) + noise).masked_fill(~mask, -1e4)
    q_dist = torch.softmax(z, -1)

    r = proper_reward(q_dist, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
    adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + loss_ce + 0.0 * act.sum()


@torch.no_grad()
def evaluate(
    model,
    tok,
    examples: list[ChoiceExample],
    *,
    max_len: int,
    head_max_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Accuracy (argmax over the masked `choice` logits vs. the gold label) on `examples`."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = _make_batch(tok, chunk, max_len, head_max_len, tok.pad_token_id)
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
        preds = logits.masked_fill(~batch["marker_mask"].to(device), -1e4).argmax(-1)
        labels = batch["label"].to(device)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    model.train(was_training)
    return correct / total if total else 0.0


def _random_baseline(examples: list[ChoiceExample]) -> float:
    if not examples:
        return 0.0
    return sum(1.0 / len(ex.criteria) for ex in examples) / len(examples)


def export_checkpoint(agent, model, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(output_dir / "model.safetensors"))

    cfg = dict(agent.cfg)
    cfg["training"] = metadata
    (output_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))

    agent.tok.save_pretrained(str(output_dir / "tokenizer"))


def run_training(cfg: TrainConfig) -> dict:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_examples = load_examples(cfg.train_dir)
    val_examples = load_examples(cfg.val_dir)
    if not train_examples:
        raise ValueError(f"no training examples found under {cfg.train_dir}")
    if not val_examples:
        raise ValueError(f"no held-out-internal examples found under {cfg.val_dir}")

    if cfg.max_train_examples is not None:
        train_examples = train_examples[: cfg.max_train_examples]
    if cfg.max_val_examples is not None:
        val_examples = val_examples[: cfg.max_val_examples]

    agent = laya.load(cfg.base_model, device=cfg.device)
    model = agent.model
    tok = agent.tok
    device = agent.device
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    model.train()

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_encoder},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    order = list(range(len(train_examples)))
    batches_per_epoch = max(1, -(-len(order) // cfg.batch_size))
    micro_steps_total = max(1, cfg.epochs * batches_per_epoch)
    optimizer_steps_total = max(1, -(-micro_steps_total // cfg.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps_total)

    start_time = time.monotonic()
    micro_step = 0
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        random.Random(cfg.seed + epoch).shuffle(order)
        for batch_start in range(0, len(order), cfg.batch_size):
            batch_indices = order[batch_start : batch_start + cfg.batch_size]
            batch_examples = [train_examples[i] for i in batch_indices]
            batch = _make_batch(tok, batch_examples, max_len, head_max_len, tok.pad_token_id)

            progress = micro_step / micro_steps_total
            sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * progress
            loss = _compute_loss(model, batch, device, sigma, cfg.group_size, cfg.w_sph, cfg.w_rps)
            (loss / cfg.grad_accum).backward()
            micro_step += 1

            if micro_step % cfg.grad_accum == 0 or micro_step == micro_steps_total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    val_accuracy = evaluate(
        model, tok, val_examples, max_len=max_len, head_max_len=head_max_len,
        batch_size=cfg.batch_size, device=device,
    )
    random_baseline = _random_baseline(val_examples)
    elapsed = time.monotonic() - start_time

    metadata = {
        "base_model": cfg.base_model,
        "dry_run": cfg.dry_run,
        "seed": cfg.seed,
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "effective_batch": cfg.batch_size * cfg.grad_accum,
            "lr_encoder": cfg.lr_encoder,
            "lr_head": cfg.lr_head,
            "weight_decay": cfg.weight_decay,
            "group_size": cfg.group_size,
            "sigma_start": cfg.sigma_start,
            "sigma_end": cfg.sigma_end,
            "w_sph": cfg.w_sph,
            "w_rps": cfg.w_rps,
        },
        "data": {
            "train_dir": str(cfg.train_dir),
            "val_dir": str(cfg.val_dir),
            "train_example_count": len(train_examples),
            "val_example_count": len(val_examples),
        },
        "held_out_internal_accuracy": val_accuracy,
        "random_baseline_accuracy": random_baseline,
        "elapsed_seconds": elapsed,
    }

    export_checkpoint(agent, model, cfg.output_dir, metadata)
    (cfg.output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _default_checkpoint_base() -> Path:
    return Path(settings.checkpoint_dir) if settings.checkpoint_dir else Path("data/checkpoints")


def _default_training_corpus_dir() -> Path:
    if settings.training_corpus_dir:
        return Path(settings.training_corpus_dir)
    return Path("data/training_corpus")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune laya's choice head on the trellis training corpus and export a "
        "safetensors checkpoint."
    )
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None, help="held_out_internal split")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--device", default=None, help="e.g. cuda, cpu (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run one epoch over a tiny slice of the data with small batches/steps, to prove "
        "the training loop is wired correctly without a real multi-hour run.",
    )
    for name in FULL_RUN_DEFAULTS:
        flag = "--" + name.replace("_", "-")
        is_int = isinstance(FULL_RUN_DEFAULTS[name], int) or name.endswith("examples")
        parser.add_argument(flag, type=int if is_int else float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    defaults = DRY_RUN_DEFAULTS if args.dry_run else FULL_RUN_DEFAULTS
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)

    training_corpus_dir = _default_training_corpus_dir()
    train_dir = args.train_dir or training_corpus_dir / "train"
    val_dir = args.val_dir or training_corpus_dir / "held_out_internal"

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_id = f"{'dryrun' if args.dry_run else 'run'}-{timestamp}"
        output_dir = _default_checkpoint_base() / run_id

    cfg = TrainConfig(
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=output_dir,
        base_model=args.base_model,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        group_size=args.group_size,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        w_sph=args.w_sph,
        w_rps=args.w_rps,
        dry_run=args.dry_run,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
    )

    metadata = run_training(cfg)
    print(json.dumps(metadata, indent=2))
    print(f"checkpoint written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
