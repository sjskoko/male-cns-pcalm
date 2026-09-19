"""Exploratory mechanism study; BP oracle arms are diagnostics, not local methods."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'credit_rebuild'))
import study as base  # noqa: E402

METHODS = ('bp', 'pc', 'alm', 'alm_bp_norm', 'bp_alm_norm', 'damped_alm')


def transfer_norm(direction, magnitude):
    """Layerwise intervention; zero direction stays zero."""
    return [d * (np.linalg.norm(m) / max(np.linalg.norm(d), 1e-30))
            for d, m in zip(direction, magnitude)]


def damped(w, x, y, budget, beta=.9):
    h, _ = base.forward(w, x)
    dual = [np.zeros_like(v) for v in h[1:]]
    scales = [np.ones(v.shape[1]) for v in h[1:]]
    for t in range(budget):
        p, _, q = base.signals(w, h, dual, scales)
        dh = [q[0] - (q[1] * (1-p[1]**2)) @ w[1].T,
              q[1] + (h[-1] @ w[-1] - y) @ w[-1].T]
        h = [x] + [v-.05*g for v, g in zip(h[1:], dh)]
        if t < budget-1:
            _, r, _ = base.signals(w, h, dual, scales)
            dual = [beta*v+.2*e for v, e in zip(dual, r)]
    p, _, q = base.signals(w, h, dual, scales)
    return [-h[i].T @ (q[i]*(1-p[i]**2))/len(x) for i in range(2)] + [
        h[-1].T @ (h[-1] @ w[-1]-y)/len(x)]


def gradients(w, x, y, masks, method, budget):
    ref = [g*m for g, m in zip(base.bp(w, x, y), masks)]
    if method == 'bp':
        return ref, ref
    if method == 'damped_alm':
        g = damped(w, x, y, budget)
    else:
        scales = [np.ones(v.shape[1]) for v in w[:-1]]
        g = base.local(w, x, y, scales, .05, 0 if method == 'pc' else .2, budget)[0]
    g = [a*m for a, m in zip(g, masks)]
    if method == 'alm_bp_norm':
        g = transfer_norm(g, ref)
    elif method == 'bp_alm_norm':
        g = transfer_norm(ref, g)
    return g, ref


def project(w, masks, signs):
    return [np.where(s != 0, s*np.maximum(a*s, 0), a)*m
            for a, m, s in zip(w, masks, signs)]


def fit(task, optimizer, seed, method, budget, epochs):
    masks, signs = base.circuit()
    x, y, xv, yv, xt, yt = base.data(masks, signs, seed, task)
    w = base.init(masks, signs, seed+30000)
    first = float(np.mean((base.forward(w, xt)[1]-yt)**2))
    mom = [np.zeros_like(a) for a in w]
    var = [np.zeros_like(a) for a in w]
    rng = np.random.default_rng(seed+40000)
    trace = []
    step = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        for batch in np.array_split(rng.permutation(len(x)), 8):
            g, ref = gradients(w, x[batch], y[batch], masks, method, budget)
            step += 1
            if optimizer == 'adam':
                mom = [.9*a+.1*b for a, b in zip(mom, g)]
                var = [.999*a+.001*b*b for a, b in zip(var, g)]
                updates = [.003*(a/(1-.9**step))/(np.sqrt(b/(1-.999**step))+1e-8)
                           for a, b in zip(mom, var)]
            else:
                updates = [.003*a for a in g]
            candidate = [a-b for a, b in zip(w, updates)]
            new = project(candidate, masks, signs)
            if step % 8 == 0:
                actual = [a-b for a, b in zip(w, new)]
                clipped = sum(np.count_nonzero((a != b) & m) for a, b, m in zip(candidate, new, masks))
                norm_ref = np.sqrt(sum(np.sum(a*a) for a in ref))
                trace.append({'epoch': epoch+1,
                    'hidden_cosine': base.cosine(g[:-1], ref[:-1], masks[:-1]),
                    'gradient_norm_ratio': np.sqrt(sum(np.sum(a*a) for a in g))/max(norm_ref, 1e-30),
                    'projected_step_cosine': base.cosine(actual, ref, masks),
                    'first_order_descent': sum(float(np.sum(a*b)) for a, b in zip(actual, ref)),
                    'clipped_fraction': clipped/sum(int(m.sum()) for m in masks),
                    'minibatch_loss_change': float(np.mean((base.forward(new, x[batch])[1]-y[batch])**2)
                                                   -np.mean((base.forward(w, x[batch])[1]-y[batch])**2))})
            w = new
        val = float(np.mean((base.forward(w, xv)[1]-yv)**2))
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite validation loss')
    row = dict(task=task, optimizer=optimizer, seed=seed, method=method, budget=budget,
               epochs=epochs, initial_mse=first, validation_mse=val,
               test_mse=float(np.mean((base.forward(w, xt)[1]-yt)**2)),
               seconds=time.perf_counter()-start, status='ok')
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(2100, 2102)) if args.profile == 'pilot' else list(range(2200, 2220))
    budgets = [8] if args.profile == 'pilot' else [4, 8, 32]
    epochs = 10 if args.profile == 'pilot' else 60
    provenance = {'profile': args.profile, 'seeds': seeds, 'budgets': budgets,
                  'epochs': epochs, 'python': platform.python_version(), 'numpy': np.__version__,
                  'code_sha256': base.digest(Path(__file__)),
                  'base_sha256': base.digest(Path(base.__file__)),
                  'circuit_sha256': base.digest(base.HERE/'circuit.npz'),
                  'protocol_sha256': base.digest(HERE/'PROTOCOL.md'),
                  'interpretation': 'Exploratory fixed-hyperparameter study, not confirmatory.'}
    (args.output/'manifest.json').write_text(json.dumps(provenance, indent=2))
    rows, traces = [], []
    for task in ('nonlinear', 'linear'):
        for optimizer in ('adam', 'sgd'):
            for seed in seeds:
                for budget in budgets:
                    for method in METHODS:
                        try:
                            row, history = fit(task, optimizer, seed, method, budget, epochs)
                        except FloatingPointError as error:
                            row = dict(task=task, optimizer=optimizer, seed=seed, budget=budget,
                                       method=method, status='failed', error=str(error))
                            history = []
                        rows.append(row)
                        traces.extend([dict(task=task, optimizer=optimizer, seed=seed,
                                            budget=budget, method=method, **h) for h in history])
                        pd.DataFrame(rows).to_csv(args.output/'trials.csv', index=False)
                        pd.DataFrame(traces).to_csv(args.output/'diagnostics.csv', index=False)
                print(task, optimizer, seed, 'completed', flush=True)
    frame = pd.DataFrame(rows)
    summary = frame[frame.status == 'ok'].groupby(['task', 'optimizer', 'budget', 'method']).test_mse.agg(['mean', 'std', 'count'])
    summary.to_csv(args.output/'summary.csv')
    (args.output/'RESULTS.md').write_text('# Exploratory results\n\n'+summary.to_markdown()+
        '\n\nFailure count: '+str(sum(frame.status != 'ok'))+
        '\n\nPilot results are feasibility evidence only. Oracle arms use BP and are not deployable local-learning proposals.\n')


if __name__ == '__main__':
    main()
