from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    DEFAULT_FEATURE_SETS,
    TensorTransitionDataset,
    apply_normalizer,
    build_iql_bundle,
    compute_normalizer,
    split_indices_by_sample_id,
)
from models import (
    IQLModelConfig,
    PolicyNet,
    QNet,
    ValueNet,
    expectile_loss,
    gather_action_values,
    soft_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline IQL SeaCache policy.")
    parser.add_argument(
        "--feature_cache",
        type=Path,
        default=Path("/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409"),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/hy-tmp/openvid_100_seacache_trace_data"),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--feature_sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument(
        "--target_speedup_offsets",
        nargs="+",
        type=float,
        default=(-0.3, -0.15, 0.0, 0.15, 0.3),
        help="Offsets around each trajectory's measured speedup used as target speedups.",
    )
    parser.add_argument("--min_target_speedup", type=float, default=1.0)
    parser.add_argument("--max_target_speedup", type=float, default=4.0)
    parser.add_argument(
        "--latent_mse_cache",
        type=Path,
        default=None,
        help="Optional cache for per-step MSE(z_{t-1}, z_gt_{t-1}).",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help="Optional torch intra-op thread count; useful for CPU smoke tests.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--target_rho", type=float, default=0.995)
    parser.add_argument("--weight_max", type=float, default=100.0)
    parser.add_argument("--lambda_latent", type=float, default=5.0)
    parser.add_argument("--lambda_recompute", type=float, default=0.04)
    parser.add_argument("--lambda_psnr", type=float, default=1.0)
    parser.add_argument("--lambda_speedup", type=float, default=30.0)
    parser.add_argument("--reuse_cost_ratio", type=float, default=0.081)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def make_loader(dataset: TensorTransitionDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(
    *,
    loader: DataLoader,
    value_net: ValueNet,
    q1_net: QNet,
    q2_net: QNet,
    target_q1: QNet,
    target_q2: QNet,
    policy_net: PolicyNet,
    value_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    value_net.train()
    q1_net.train()
    q2_net.train()
    policy_net.train()
    totals: dict[str, float] = {
        "v_loss": 0.0,
        "q_loss": 0.0,
        "pi_loss": 0.0,
        "skip_rate_data": 0.0,
        "skip_rate_policy": 0.0,
        "reward": 0.0,
    }
    examples = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        state = batch["state"].float()
        next_state = batch["next_state"].float()
        action = batch["action"].long()
        reward = batch["reward"].float()
        done = batch["done"].float()
        batch_size = int(state.shape[0])

        with torch.no_grad():
            tq1 = gather_action_values(target_q1(state), action)
            tq2 = gather_action_values(target_q2(state), action)
            q_target = torch.minimum(tq1, tq2)
        v_pred = value_net(state)
        v_loss = expectile_loss(q_target - v_pred, args.tau)
        value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        value_opt.step()

        with torch.no_grad():
            y = reward + args.gamma * (1.0 - done) * value_net(next_state)
        q1_pred = gather_action_values(q1_net(state), action)
        q2_pred = gather_action_values(q2_net(state), action)
        q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()
        soft_update(target_q1, q1_net, args.target_rho)
        soft_update(target_q2, q2_net, args.target_rho)

        with torch.no_grad():
            q_min = torch.minimum(
                gather_action_values(q1_net(state), action),
                gather_action_values(q2_net(state), action),
            )
            advantage = q_min - value_net(state)
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
            weights = torch.exp(args.beta * advantage).clamp(max=args.weight_max)
        logits = policy_net(state)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.view(-1, 1)).squeeze(1)
        pi_loss = -(weights * log_prob).mean()
        policy_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        policy_opt.step()

        with torch.no_grad():
            policy_action = logits.argmax(dim=-1)
        for key, value in (
            ("v_loss", v_loss.item()),
            ("q_loss", q_loss.item()),
            ("pi_loss", pi_loss.item()),
            ("skip_rate_data", action.float().mean().item()),
            ("skip_rate_policy", policy_action.float().mean().item()),
            ("reward", reward.mean().item()),
        ):
            totals[key] += value * batch_size
        examples += batch_size
        if args.log_every and (batch_index + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "v_loss": round(v_loss.item(), 6),
                        "q_loss": round(q_loss.item(), 6),
                        "pi_loss": round(pi_loss.item(), 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {key: value / max(examples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    *,
    loader: DataLoader,
    value_net: ValueNet,
    q1_net: QNet,
    q2_net: QNet,
    policy_net: PolicyNet,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    value_net.eval()
    q1_net.eval()
    q2_net.eval()
    policy_net.eval()
    totals: dict[str, float] = {
        "v_loss": 0.0,
        "q_loss": 0.0,
        "pi_loss": 0.0,
        "policy_accuracy": 0.0,
        "skip_rate_data": 0.0,
        "skip_rate_policy": 0.0,
        "reward": 0.0,
    }
    examples = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        state = batch["state"].float()
        next_state = batch["next_state"].float()
        action = batch["action"].long()
        reward = batch["reward"].float()
        done = batch["done"].float()
        batch_size = int(state.shape[0])

        q1_action = gather_action_values(q1_net(state), action)
        q2_action = gather_action_values(q2_net(state), action)
        q_min = torch.minimum(q1_action, q2_action)
        v_pred = value_net(state)
        v_loss = expectile_loss(q_min - v_pred, args.tau)
        y = reward + args.gamma * (1.0 - done) * value_net(next_state)
        q_loss = F.mse_loss(q1_action, y) + F.mse_loss(q2_action, y)
        advantage = q_min - v_pred
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        weights = torch.exp(args.beta * advantage).clamp(max=args.weight_max)
        logits = policy_net(state)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.view(-1, 1)).squeeze(1)
        pi_loss = -(weights * log_prob).mean()
        policy_action = logits.argmax(dim=-1)
        accuracy = (policy_action == action).float().mean()

        for key, value in (
            ("v_loss", v_loss.item()),
            ("q_loss", q_loss.item()),
            ("pi_loss", pi_loss.item()),
            ("policy_accuracy", accuracy.item()),
            ("skip_rate_data", action.float().mean().item()),
            ("skip_rate_policy", policy_action.float().mean().item()),
            ("reward", reward.mean().item()),
        ):
            totals[key] += value * batch_size
        examples += batch_size
    return {key: value / max(examples, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    value_net: ValueNet,
    q1_net: QNet,
    q2_net: QNet,
    target_q1: QNet,
    target_q2: QNet,
    policy_net: PolicyNet,
    normalizer: dict[str, torch.Tensor],
    model_config: IQLModelConfig,
    args: argparse.Namespace,
    dataset_manifest: dict[str, object],
    metrics: dict[str, object],
) -> None:
    torch.save(
        {
            "value_net": value_net.state_dict(),
            "q1_net": q1_net.state_dict(),
            "q2_net": q2_net.state_dict(),
            "target_q1": target_q1.state_dict(),
            "target_q2": target_q2.state_dict(),
            "policy_net": policy_net.state_dict(),
            "normalizer": normalizer,
            "model_config": asdict(model_config),
            "train_config": vars(args),
            "dataset_manifest": dataset_manifest,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    bundle = build_iql_bundle(
        feature_cache=args.feature_cache,
        data_root=args.data_root,
        feature_sets=args.feature_sets,
        target_speedup_offsets=args.target_speedup_offsets,
        min_target_speedup=args.min_target_speedup,
        max_target_speedup=args.max_target_speedup,
        max_examples=args.max_examples,
        lambda_latent=args.lambda_latent,
        lambda_recompute=args.lambda_recompute,
        lambda_psnr=args.lambda_psnr,
        lambda_speedup=args.lambda_speedup,
        reuse_cost_ratio=args.reuse_cost_ratio,
        latent_mse_cache=(
            args.latent_mse_cache
            if args.latent_mse_cache is not None
            else args.feature_cache / "latent_mse_to_baseline.pt"
        ),
    )
    train_indices, val_indices = split_indices_by_sample_id(
        bundle.sample_ids,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    normalizer = compute_normalizer(bundle.states, train_indices)
    bundle.states = apply_normalizer(bundle.states, normalizer)
    bundle.next_states = apply_normalizer(bundle.next_states, normalizer)
    with (args.out_dir / "dataset_manifest.json").open("w") as handle:
        json.dump(bundle.manifest, handle, indent=2, default=str)
    with (args.out_dir / "split.json").open("w") as handle:
        json.dump(
            {
                "train_count": len(train_indices),
                "val_count": len(val_indices),
                "train_fraction": args.train_fraction,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    train_loader = make_loader(TensorTransitionDataset(bundle, train_indices), args, shuffle=True)
    val_loader = make_loader(TensorTransitionDataset(bundle, val_indices), args, shuffle=False)

    device = torch.device(args.device)
    model_config = IQLModelConfig(
        input_dim=int(bundle.states.shape[1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    value_net = ValueNet(model_config).to(device)
    q1_net = QNet(model_config).to(device)
    q2_net = QNet(model_config).to(device)
    target_q1 = QNet(model_config).to(device)
    target_q2 = QNet(model_config).to(device)
    policy_net = PolicyNet(model_config).to(device)
    target_q1.load_state_dict(q1_net.state_dict())
    target_q2.load_state_dict(q2_net.state_dict())

    value_opt = torch.optim.AdamW(value_net.parameters(), lr=args.lr)
    q_opt = torch.optim.AdamW(list(q1_net.parameters()) + list(q2_net.parameters()), lr=args.lr)
    policy_opt = torch.optim.AdamW(policy_net.parameters(), lr=args.lr)

    best_val_policy_accuracy = float("-inf")
    all_metrics: list[dict[str, object]] = []
    metrics_path = args.out_dir / "epoch_metrics.jsonl"
    t0 = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            loader=train_loader,
            value_net=value_net,
            q1_net=q1_net,
            q2_net=q2_net,
            target_q1=target_q1,
            target_q2=target_q2,
            policy_net=policy_net,
            value_opt=value_opt,
            q_opt=q_opt,
            policy_opt=policy_opt,
            args=args,
            device=device,
        )
        val_metrics = evaluate(
            loader=val_loader,
            value_net=value_net,
            q1_net=q1_net,
            q2_net=q2_net,
            policy_net=policy_net,
            args=args,
            device=device,
        )
        row = {
            "epoch": epoch,
            "elapsed_seconds": round(perf_counter() - t0, 3),
            "train": train_metrics,
            "val": val_metrics,
        }
        all_metrics.append(row)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["policy_accuracy"] > best_val_policy_accuracy:
            best_val_policy_accuracy = val_metrics["policy_accuracy"]
            save_checkpoint(
                args.out_dir / "best_model.pt",
                value_net=value_net,
                q1_net=q1_net,
                q2_net=q2_net,
                target_q1=target_q1,
                target_q2=target_q2,
                policy_net=policy_net,
                normalizer=normalizer,
                model_config=model_config,
                args=args,
                dataset_manifest=bundle.manifest,
                metrics=row,
            )

    final_metrics = {
        "best_val_policy_accuracy": best_val_policy_accuracy,
        "best_checkpoint_metric": "val.policy_accuracy",
        "epochs": args.epochs,
        "last": all_metrics[-1] if all_metrics else None,
    }
    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(final_metrics, handle, indent=2)
    save_checkpoint(
        args.out_dir / "final_model.pt",
        value_net=value_net,
        q1_net=q1_net,
        q2_net=q2_net,
        target_q1=target_q1,
        target_q2=target_q2,
        policy_net=policy_net,
        normalizer=normalizer,
        model_config=model_config,
        args=args,
        dataset_manifest=bundle.manifest,
        metrics=final_metrics,
    )


if __name__ == "__main__":
    main()
