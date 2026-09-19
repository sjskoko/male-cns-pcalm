# Revised staged study — 2026-09-19

## What changed
Previous fixed-weight diagnostics established numerical correctness, not learning improvement. The next task is a controlled **development-stage learning study**, plus reproduction of an unmodified official PC-ALM reference cell. This is not the confirmatory study previously gated on novelty and functional flyvis reproduction. Those gates remain mandatory for a new-method/biological task paper. No new method is claimed in this pilot.

Research topic: **Conditions for accurate and useful PC-ALM learning under measured MaleCNS wiring constraints.**
RQ1: reproduce the official reference; RQ2: does recurrent PC-ALM actually learn held-out labels on the measured graph? RQ3: does the measured feedback-direction restriction alter learning at equal iteration budgets? RQ4: does learned recurrent connectivity improve on a readout-only baseline? RQ5 (future): can a novel, cost-matched feedback remedy improve both credit and task performance?

## References / limits on novelty
Seely & Gould, PC-ALM (2026): https://arxiv.org/abs/2605.31022 ; code https://github.com/SakanaAI/pc-alm at 660747f61a8a7e547c0ecd2c48c8883380a7d1f6 .
Millidge et al., arbitrary computation graphs: https://arxiv.org/abs/2006.04182 .
Liao et al., weight symmetry (2016): https://ojs.aaai.org/index.php/AAAI/article/view/10279 .
Millidge et al., temporal PC (2024): https://doi.org/10.1371/journal.pcbi.1011183 .
Wang et al., ALM RNN (2025): https://doi.org/10.1137/23M1627614 .
Lappalainen et al., flyvis (2024): https://doi.org/10.1038/s41586-024-07939-3 ; https://github.com/TuragaLab/flyvis .
Data https://male-cns.janelia.org/download/ and sklearn load_digits (UCI handwritten digits): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html .
No novelty claimed for recurrent ALM, asymmetry, transpose recovery, or spectral projection. Candidate contribution remains measured anatomy/communication constraints plus a future validated remedy. Literature does not justify predicting a task accuracy advantage.

## Gate A: official reference, unmodified code
Run upstream tests, then Fashion-MNIST full 60000/10000, width32/depth32 ReLU seed0, 1 epoch, T=64, official frozen state-LR table, all BP/PC/PCALM. No quick/synthetic substitution. Upstream reported accuracies: .7866/.6813/.7775. Predeclared reproduction gate: each within 0.02 absolute accuracy and PCALM>PC. This is a numerical tolerance, not a statistical equivalence test. Record exact environment/commit/data hashes and all logs. If failure, diagnose before interpreting downstream as corroboration of PCALM.

## Gate B: learnability and feedback pilot, no new-method claim
Reuse threshold>=5 synapses, 132 nodes/2663 edges from verified feedback_audit; check raw data hash and derived edge hash match audit. All edges retained. No graph selection based on current outcomes. One biological specimen/subgraph; seeds are optimization variation, not animal replication.

Data: real 8x8 handwritten digits (1797), x scaled to [0,1], 10-class one-hot labels. Stratified fixed split seed45100: train900, validation300, test597. Persist indices and dataset hash. These are real images but NOT natural fly stimuli, retinal mapping or physiology. Test labels never used for fitting, tuning or epoch selection; pilot held-out estimates are still not independent future confirmation.

Column convention: h=(I-W)^(-1) Bx. B is a fixed seeded dense random projection (132x64, variance1/64), external input to all nodes; an explicit engineering choice to remove unknown retinal mapping from algorithm validation. This changes the earlier artificial motion task. Linear steady-state model, not a nonlinear fly model or time-unrolled network. C is 10x132, trainable only at the 24 preselected descending-neuron columns. Loss .5*mean_b ||Ch-y||². W respects measured support S and known source signs. Unknown signs remain unconstrained. W initialized signed log-synapse-scaled random magnitude, spectral norm .4. Enforce ||W||_2<=.8 after every update; enforce ||C||_2<=2 after every update. These global projections/linear solves are oracle engineering controls, NOT local/biologically realistic computation. Account wall time including them.

A=I-W, r=Ah-Bx. Exact implicit reference lambda*=-A^-T C^T(Ch*-y); gradient_W=-mean_b lambda* h*^T, gradient_C=mean_b(Ch*-y)h*^T, both masks applied. Compare:
- implicit: exact differentiation of equilibrium (not BPTT)
- pc: local inference, dual=0
- alm: exact transpose feedback
- masked: ALM surrogate replacing A^T by F=I-(W^T elementwise S)
- readout: W frozen, C only updated with exact gradient.
F only enforces directed support; transmitter sign of auxiliary feedback NOT constrained. Missing reverse links below threshold or outside subset are not evidence they do not exist in the whole animal.

Initialize free h at the solved forward equilibrium each batch, lambda=0. Inference h+=h-eta[C^T(Ch-y)+F(lambda+r)], then lambda+=lambda+alpha*r(h+). eta=.1, alpha=.2 for ALM/surrogate, PC alpha=0; rho=1. Credit uses pre-final-dual multiplier (official convention). T=32 for ALL local methods. Forward equilibrium and auxiliary matrices have equal implementation across paired methods. No claim matched BP FLOPs. Dense implementation timings not sparse hardware efficiency.

Fit Adam, 20 epochs, batch150, final epoch only. Weight and readout common LR chosen separately for each method from {.001,.003} using 3 development seeds46000..46002, lowest mean validation loss. Every method gets same grid and paired data/initialization/batch order. Then 5 fresh evaluation seeds47000..47004 with each chosen LR; single test evaluation at end. No checkpoint tuning or retries based on test. Total30 development+25 evaluation fits. Persist every failed attempt and exclude no seed. Divergence is a failure, not a silently dropped observation.
Primary: test half-squared error. Secondary: test accuracy, initial-vs-final validation loss, W update norm, gradient relative error on fixed training probe, constraint residual, wall time. Readout-only control tests whether circuit learning adds anything at all. Compare ALM-PC, ALM-implicit, masked-ALM, ALM-readout using paired raw differences. With only5 seeds give all observations, mean/SD and exploratory t intervals; no significance/novelty claims. Confirmatory seed count/power requires subsequent protocol.

Gate B pass: exact reference and ALM each achieve mean validation accuracy>=.70 and >=20% relative validation-loss reduction in fresh seeds. This is an engineering feasibility threshold, not predicted effect. To advance a claim of useful circuit learning, ALM must also improve over readout-only mean validation loss by>=1%; otherwise output-layer confounding remains. To advance a remedy claim requires a separate intervention at equal communication budget, absent in this pilot.

## Verification before fitting
Finite differences of masked W,C gradients; alpha=0 local gradient matches PC; masking does not change forward graph; split disjointness; sign/support/spectral projection invariants. Save protocol/code/data hashes before fitting. Checks are independent numerical properties, not only implementation mirroring.

## Predictions and stop rules
Expected: implicit reference should learn if this simplified adapter/task is usable. ALM can approximate the implicit gradient with adequate inference; T32 may be insufficient as W,C evolve. Masked surrogate can remain biased, but task impact may be small because cosine was near1 in prior diagnostic. Thus ALM superiority is NOT assumed; readout-only may match or win. If reference fails gate, diagnose task/optimizer; if ALM fails with reference success, inspect inference budget/stability; if readout matches, do not claim useful internal learning.

## What remains before confirmatory research
Official reference reproduction, nonlinear functional flyvis reproduction and biologically grounded input; explicit nearest-prior-art comparison for one remedy; development budget sweep32/64/128 with equal accounting, gradient decomposition and capacity controls; source-recording-level video splits; separate three-circuit/20-seed confirmation with frozen effect-size/precision plan. Real task performance and clear novelty cannot be guaranteed beforehand. Failures remain publishable records, not proof the fly brain is useless.
