"""Small live training demonstration; never modifies the published study."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/credit_rebuild'))
import study  # noqa: E402


def run_demo():
    masks, signs = study.circuit()
    results = {}
    for method in ('bp', 'pc', 'pcalm'):
        row, history, _ = study.fit(
            masks, signs, 'nonlinear', 20260919, method,
            (.003, .05, .2 if method == 'pcalm' else 0.),
            epochs=20, final=True,
        )
        results[method] = {'metrics': row, 'history': history}
        print(f"{method:6s}: initial test MSE={row['initial_test_mse']:.5f}, "
              f"final={row['test_mse']:.5f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, result in results.items():
        ax.plot([h['epoch'] for h in result['history']],
                [h['validation_mse'] for h in result['history']], label=method.upper())
    ax.set(xlabel='Epoch', ylabel='Validation MSE',
           title='Live demo: one seed, synthetic teacher, real projected wiring')
    ax.legend()
    fig.tight_layout()
    return results, fig


if __name__ == '__main__':
    results, fig = run_demo()
    out = ROOT / 'examples/demo_output'
    out.mkdir(exist_ok=True)
    (out / 'demo_results.json').write_text(json.dumps(results, indent=2))
    fig.savefig(out / 'learning.png', dpi=160)
    print('Demonstration only; not a replacement for the 760-fit study.')
