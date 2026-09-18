# Literature boundary and candidate contribution

Checked primary sources:

- Seely & Gould (2026), [Augmented Lagrangian Predictive Coding](https://arxiv.org/abs/2605.31022).
  The base PC-ALM algorithm and its local multiplier updates are prior work.
  Its reparameterization experiments already address sensitivity to conditioning.
- Salvatori et al. (2022), [Learning on Arbitrary Graph Topologies via Predictive Coding](https://arxiv.org/abs/2201.13180).
  Predictive coding on nonstandard graphs is not itself a new contribution.
- Zahid et al. (2023), [Sample as You Infer: Predictive Coding With Langevin Dynamics](https://arxiv.org/abs/2311.13664).
  Preconditioning in predictive coding also predates this experiment, although
  its stochastic generative-model setting differs from the present supervised ALM.

Potential contribution: an audited connectome-grounded evaluation of finite-budget
PC-ALM and a local degree-scaled constraint variant, compared against a within-layer
shuffled-scale control with equal tuning counts. This is a candidate contribution,
not an exhaustive prior-art review or a claim to be the first such method.

Evidence must distinguish (1) correct local updates, (2) successful task learning,
(3) better gradient alignment, (4) lower held-out error, and (5) degree-specific
benefit. None of these logically implies all the others. A negative performance
result should revise the scientific claim, not trigger test-set tuning.

This study alone is insufficient for a strong neuroscience paper: it uses one
feedforward graph projection and synthetic targets. A broader machine-learning
paper needs more circuits, inference budgets during training, matched parameter
controls, generic preconditioners, and larger development searches. Recurrent
connectome learning and measured targets remain untested.
