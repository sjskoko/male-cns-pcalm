import numpy as np

from run import base, damped, project, transfer_norm


def test_damping_one_matches_alm():
    masks, signs = base.circuit()
    w = base.init(masks, signs, 99)
    x, y, *_ = base.data(masks, signs, 99, 'nonlinear')
    scales = [np.ones(v.shape[1]) for v in w[:-1]]
    for budget in (1, 8, 32):
        ref = base.local(w, x[:8], y[:8], scales, .05, .2, budget)[0]
        for a, b in zip(ref, damped(w, x[:8], y[:8], budget, beta=1)):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_norm_intervention_preserves_direction():
    a = np.array([[1., -2.], [0., 3.]])
    b = np.array([[4., 1.], [0., -2.]])
    g = transfer_norm([a], [b])[0]
    np.testing.assert_allclose(np.linalg.norm(g), np.linalg.norm(b))
    np.testing.assert_allclose(g/np.linalg.norm(g), a/np.linalg.norm(a))
    np.testing.assert_array_equal(transfer_norm([a*0], [b])[0], a*0)


def test_projection_feasibility_and_idempotence():
    w = [np.array([[-1., 7.], [3., -4.]])]
    masks = [np.array([[True, False], [True, True]])]
    signs = [np.array([[1, 0], [-1, 0]])]
    p = project(w, masks, signs)
    np.testing.assert_array_equal(p[0], [[0, 0], [0, -4]])
    np.testing.assert_array_equal(project(p, masks, signs)[0], p[0])
