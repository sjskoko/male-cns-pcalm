"""Temporal lifted-state PC/ALM on measured MaleCNS edges."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph(kind):
    nodes = pd.read_csv(ROOT/'experiments/malecns_v1_real/assignments.csv').sort_values('body_id')
    edges = pd.read_csv(ROOT/'experiments/credit_rebuild/all_selected_edges.csv')
    ids = nodes.body_id.to_numpy()
    syn = np.zeros((len(ids), len(ids)))
    if kind == 'forward':
        edges = edges[edges.post_layer == edges.pre_layer+1]
    syn[np.searchsorted(ids, edges.pre), np.searchsorted(ids, edges.post)] = edges.synapses
    mask = syn > 0
    sign = nodes.sign.to_numpy()[:, None]
    b = np.zeros((32, len(ids)))
    b[np.arange(32), np.where(nodes.layer.to_numpy() == 0)[0]] = 1
    out = np.where(nodes.layer.to_numpy() == 3)[0]
    return mask, sign, b, out, syn


def dataset(seed):
    images = load_digits().images/16.
    split = np.random.default_rng(731).permutation(len(images))
    pools = [split[:1000], split[1000:1400], split[1400:]]
    result = []
    for k, (pool, count) in enumerate(zip(pools, [128, 64, 128])):
        rng = np.random.default_rng(seed+1000*k)
        chosen = rng.choice(pool, count, replace=False)
        labels = np.tile(np.arange(4), count//4)
        rng.shuffle(labels)
        velocity = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])[labels]
        xs = []
        for idx, (dx, dy) in zip(chosen, velocity):
            offset = rng.integers(0, 8, size=2)
            frames = [np.roll(images[idx], (int(offset[0]+t*dy), int(offset[1]+t*dx)), axis=(0, 1))
                      .reshape(4, 2, 8).mean(1).ravel() for t in range(6)]
            xs.append(frames)
        result.append((np.asarray(xs).transpose(1, 0, 2), velocity.astype(float), chosen))
    return result


def forward(w, c, x, b, out):
    h = [np.zeros((x.shape[1], len(w)))]
    for xt in x:
        h.append(.5*h[-1]+.5*np.tanh(h[-1]@w+xt@b))
    return h, h[-1][:, out]@c


def bp(w, c, x, y, b, out):
    h, prediction = forward(w, c, x, b, out)
    delta = prediction-y
    gc = h[-1][:, out].T@delta/len(y)
    dh = np.zeros_like(h[-1])
    dh[:, out] = delta@c.T
    gw = np.zeros_like(w)
    for t in range(len(x)-1, -1, -1):
        p = np.tanh(h[t]@w+x[t]@b)
        z = .5*dh*(1-p*p)
        gw += h[t].T@z/len(y)
        dh = .5*dh+z@w.T
    return [gw, gc]


def local(w, c, x, y, b, out, alpha, steps=8):
    h, _ = forward(w, c, x, b, out)
    dual = [np.zeros_like(v) for v in h[1:]]
    def signals(states):
        p = [np.tanh(states[t]@w+x[t]@b) for t in range(len(x))]
        r = [states[t+1]-.5*states[t]-.5*p[t] for t in range(len(x))]
        return p, r, [d+e for d, e in zip(dual, r)]
    for step in range(steps):
        p, _, q = signals(h)
        grad = []
        for t in range(len(x)):
            g = q[t].copy()
            if t+1 < len(x):
                g -= .5*q[t+1]+(.5*q[t+1]*(1-p[t+1]**2))@w.T
            else:
                g[:, out] += (h[-1][:, out]@c-y)@c.T
            grad.append(g)
        h = [h[0]]+[v-.05*g for v, g in zip(h[1:], grad)]
        if step < steps-1:
            _, r, _ = signals(h)
            dual = [d+alpha*e for d, e in zip(dual, r)]
    p, r, q = signals(h)
    gw = sum(-h[t].T@(.5*q[t]*(1-p[t]**2))/len(y) for t in range(len(x)))
    gc = h[-1][:, out].T@(h[-1][:, out]@c-y)/len(y)
    return [gw, gc], float(np.sqrt(np.mean(np.square(r))))


def metrics(w, c, x, y, b, out):
    pred = forward(w, c, x, b, out)[1]
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    return float(np.mean((pred-y)**2)), float(np.mean(
        np.argmin(((pred[:, None]-directions)**2).sum(2), axis=1) ==
        np.argmin(((y[:, None]-directions)**2).sum(2), axis=1)))


def fit(seed, kind, method, epochs):
    mask, signs, b, out, syn = graph(kind)
    rng = np.random.default_rng(seed+9000)
    w = rng.normal(size=mask.shape)*np.log1p(syn)
    w = np.where(signs != 0, np.abs(w)*signs, w)*mask
    # Conservative initial gain, not a bound maintained throughout training.
    w *= .8/max(np.linalg.norm(w, 2), 1e-12)
    c = rng.normal(size=(len(out), 2))*.1
    (x, y, _), (xv, yv, _), (xt, yt, _) = dataset(seed)
    initial = metrics(w, c, xt, yt, b, out)[0]
    mom = [np.zeros_like(w), np.zeros_like(c)]
    var = [v.copy() for v in mom]
    step = 0
    curves = []
    start = time.perf_counter()
    for epoch in range(epochs):
        order = np.random.default_rng(seed+epoch+50000).permutation(len(y))
        for batch in np.array_split(order, 4):
            if method == 'bp':
                g = bp(w, c, x[:, batch], y[batch], b, out)
                residual = 0.
            else:
                g, residual = local(w, c, x[:, batch], y[batch], b, out, .2 if method == 'alm' else 0.)
            g[0] *= mask
            step += 1
            for i, a in enumerate([w, c]):
                mom[i] = .9*mom[i]+.1*g[i]
                var[i] = .999*var[i]+.001*g[i]**2
                a -= .003*(mom[i]/(1-.9**step))/(np.sqrt(var[i]/(1-.999**step))+1e-8)
            w = np.where(signs != 0, signs*np.maximum(w*signs, 0), w)*mask
        loss = metrics(w, c, xv, yv, b, out)[0]
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        curves.append(dict(epoch=epoch+1, validation_mse=loss, residual=residual))
    mse, acc = metrics(w, c, xt, yt, b, out)
    occluded = xt.copy()
    occluded[2:4] = 0
    shuffled = xt[np.random.default_rng(seed+888).permutation(len(xt))]
    row = dict(seed=seed, graph=kind, method=method, edges=int(mask.sum()), initial_mse=initial,
               test_mse=mse, accuracy=acc, occluded_mse=metrics(w, c, occluded, yt, b, out)[0],
               shuffled_mse=metrics(w, c, shuffled, yt, b, out)[0],
               seconds=time.perf_counter()-start, status='ok')
    return row, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), HERE/'PROTOCOL.md',
        ROOT/'experiments/credit_rebuild/all_selected_edges.csv',
        ROOT/'experiments/malecns_v1_real/assignments.csv']}
    manifest = dict(hashes=provenance, sklearn=sklearn.__version__, numpy=np.__version__,
                    image_sha256=hashlib.sha256(load_digits().images.tobytes()).hexdigest(),
                    seeds=list(range(3100, 3100+args.seeds)), epochs=args.epochs,
                    input='Real digit images, synthetic periodic translation; not natural video')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, history = [], []
    for seed in manifest['seeds']:
        for kind in ('forward', 'full'):
            for method in ('bp', 'pc', 'alm'):
                try:
                    row, curve = fit(seed, kind, method, args.epochs)
                except FloatingPointError as exc:
                    row = dict(seed=seed, graph=kind, method=method, status='failed', error=str(exc))
                    curve = []
                rows.append(row)
                history.extend([dict(seed=seed, graph=kind, method=method, **h) for h in curve])
                pd.DataFrame(rows).to_csv(args.output/'trials.csv', index=False)
                pd.DataFrame(history).to_csv(args.output/'curves.csv', index=False)
        print(seed, 'complete', flush=True)
    df = pd.DataFrame(rows)
    summary = df[df.status == 'ok'].groupby(['graph', 'method'])[
        ['test_mse', 'accuracy', 'occluded_mse', 'shuffled_mse']].agg(['mean', 'std'])
    summary.to_csv(args.output/'summary.csv')
    comparisons = []
    for kind in ('forward', 'full'):
        p = df[(df.graph == kind) & (df.status == 'ok')].pivot(index='seed', columns='method', values='test_mse')
        d = (p.pc-p.alm).dropna().to_numpy()
        rng = np.random.default_rng(883)
        ci = np.quantile(rng.choice(d, (10000, len(d)), replace=True).mean(1), [.025, .975])
        comparisons.append(dict(graph=kind, paired_n=len(d), pc_minus_alm=float(d.mean()),
                                ci95=ci.tolist(), p=float(wilcoxon(d).pvalue) if np.any(d) else 1.))
    order = np.argsort([c['p'] for c in comparisons])
    adjusted = 0.
    for rank, i in enumerate(order):
        adjusted = max(adjusted, min(1., (len(order)-rank)*comparisons[i]['p']))
        comparisons[i]['holm_p'] = adjusted
    (args.output/'comparisons.json').write_text(json.dumps(comparisons, indent=2))
    (args.output/'RESULTS.md').write_text('# Temporal feasibility results\n\n'+summary.to_markdown()+
        '\n\nPC minus ALM (positive favors ALM):\n\n```json\n'+json.dumps(comparisons, indent=2)+
        '\n```\n\nFailure count: '+str(sum(df.status != 'ok'))+
        '\n\nExploratory, fixed hyperparameters; not a confirmatory biological result.\n')


if __name__ == '__main__':
    main()
