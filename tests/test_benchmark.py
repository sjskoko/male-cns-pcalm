from __future__ import annotations

from flypcalm.benchmark import run_visual_motor_benchmark, write_benchmark_results


def test_full_virtual_matrix_writes_expected_artifacts(tmp_path):
    result = run_visual_motor_benchmark(
        {
            "device": "cpu",
            "synthetic_graph": "visual_proxy",
            "layer_sizes": [4, 6, 4],
            "density": 0.5,
            "topologies": ["native", "degree_preserving", "random"],
            "methods": ["bp", "pc", "pcalm"],
            "seeds": [0],
            "train_samples": 32,
            "test_samples": 16,
            "batch_size": 32,
            "epochs": 1,
            "budget": 1,
            "state_lr": 0.02,
            "alpha": 0.25,
        }
    )
    assert result["biological_claim"] is False
    assert len(result["trials"]) == 9
    assert len(result["aggregate"]) == 9
    assert all("test_accuracy_auc" in trial for trial in result["trials"])
    output = tmp_path / "benchmark"
    write_benchmark_results(result, output)
    for filename in ("result.json", "report.md", "trials.csv", "metrics.csv", "aggregate.csv"):
        assert (output / filename).is_file()
    assert "not a MaleCNS result" in (output / "report.md").read_text(encoding="utf-8")
