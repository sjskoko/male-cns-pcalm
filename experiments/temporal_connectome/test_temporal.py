import numpy as np

from experiment import bp, dataset, forward, graph, local


def fixture():
    rng = np.random.default_rng(42)
    return (rng.normal(size=(3, 3))*.1, rng.normal(size=(2, 2))*.1,
            rng.normal(size=(4, 2, 2)), rng.normal(size=(2, 2)),
            rng.normal(size=(2, 3))*.1, np.array([1, 2]))


def test_bptt_finite_difference():
    w, c, x, y, b, out = fixture()
    gradients = bp(w, c, x, y, b, out)
    for value, gradient in zip([w, c], gradients):
        for idx in np.ndindex(value.shape):
            old = value[idx]
            value[idx] = old+1e-6
            plus = .5*np.sum((forward(w, c, x, b, out)[1]-y)**2)/len(y)
            value[idx] = old-1e-6
            minus = .5*np.sum((forward(w, c, x, b, out)[1]-y)**2)/len(y)
            value[idx] = old
            np.testing.assert_allclose(gradient[idx], (plus-minus)/2e-6, atol=1e-8)


def test_first_inference_step_equivalence():
    w, c, x, y, b, out = fixture()
    a = local(w, c, x, y, b, out, 0., steps=1)[0]
    d = local(w, c, x, y, b, out, .2, steps=1)[0]
    for u, v in zip(a, d):
        np.testing.assert_array_equal(u, v)


def test_measured_graph_and_disjoint_sources():
    assert graph('full')[0].sum() == 2663
    assert graph('forward')[0].sum() == 1001
    parts = dataset(3100)
    for i in range(3):
        for j in range(i):
            assert not set(parts[i][2]) & set(parts[j][2])
    for _, y, _ in parts:
        np.testing.assert_allclose(np.mean(y*y), .5)
