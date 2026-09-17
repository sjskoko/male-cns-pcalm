from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .benchmark import run_visual_motor_benchmark, write_benchmark_results
from .data import build_layered_connectome
from .experiment import run_experiment, write_results
from .malecns import select_visual_descending_circuit, write_circuit_selection


def _select_malecns(args: argparse.Namespace) -> None:
    assignments, report = select_visual_descending_circuit(
        args.annotations,
        args.neurotransmitters,
        args.weights,
        layer_sizes=tuple(args.layer_sizes),
        min_weight=args.min_weight,
        nt_confidence=args.nt_confidence,
    )
    write_circuit_selection(assignments, report, args.output, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


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


def _benchmark(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    result = run_visual_motor_benchmark(config)
    write_benchmark_results(result, args.output_dir)
    print(json.dumps(result["aggregate"], indent=2))


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

    select_malecns = subcommands.add_parser(
        "select-malecns", help="derive a visual-to-descending circuit from MaleCNS v1.0"
    )
    select_malecns.add_argument("--annotations", required=True)
    select_malecns.add_argument("--neurotransmitters", required=True)
    select_malecns.add_argument("--weights", required=True)
    select_malecns.add_argument("--output", required=True, help="output assignments CSV")
    select_malecns.add_argument("--report", required=True, help="output selection JSON")
    select_malecns.add_argument(
        "--layer-sizes", nargs=4, type=int, default=[32, 48, 40, 24]
    )
    select_malecns.add_argument("--min-weight", type=float, default=5.0)
    select_malecns.add_argument("--nt-confidence", type=float, default=0.5)
    select_malecns.set_defaults(func=_select_malecns)

    train = subcommands.add_parser("train", help="run a configured experiment")
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir", default="results/run")
    train.set_defaults(func=_train)

    smoke = subcommands.add_parser("smoke", help="run a small synthetic end-to-end check")
    smoke.add_argument("--method", choices=["bp", "pc", "pcalm"], default="pcalm")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(func=_smoke)

    benchmark = subcommands.add_parser(
        "benchmark", help="run topology x learning-rule visual-motor experiments"
    )
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output-dir", default="results/visual-motor")
    benchmark.set_defaults(func=_benchmark)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
