from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import build_layered_connectome
from .experiment import run_experiment, write_results


def _prepare(args: argparse.Namespace) -> None:
    connectome, report = build_layered_connectome(
        args.edges,
        args.assignments,
        min_weight=args.min_weight,
        report_path=args.report,
    )
    connectome.save(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _train(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    result = run_experiment(config)
    write_results(result, args.output_dir)
    print(json.dumps(result["final"], indent=2))


def _smoke(args: argparse.Namespace) -> None:
    result = run_experiment(
        {
            "method": args.method,
            "device": args.device,
            "seed": args.seed,
            "layer_sizes": [8, 12, 10],
            "output_dim": 3,
            "samples": 96,
            "epochs": 3,
            "batch_size": 24,
            "density": 0.4,
            "budget": 5,
            "state_lr": 0.05,
            "alpha": 0.5,
            "learning_rate": 0.005,
        }
    )
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="flypcalm")
    subcommands = root.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="project MaleCNS edges into ordered layers")
    prepare.add_argument("--edges", required=True, help="MaleCNS weights feather/parquet/csv")
    prepare.add_argument("--assignments", required=True, help="body_id/layer/module/sign table")
    prepare.add_argument("--output", required=True, help="output .pt connectome")
    prepare.add_argument("--report", default="results/projection-report.json")
    prepare.add_argument("--min-weight", type=float, default=5.0)
    prepare.set_defaults(func=_prepare)

    train = subcommands.add_parser("train", help="run a configured experiment")
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir", default="results/run")
    train.set_defaults(func=_train)

    smoke = subcommands.add_parser("smoke", help="run a small synthetic end-to-end check")
    smoke.add_argument("--method", choices=["bp", "pc", "pcalm"], default="pcalm")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(func=_smoke)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
