# Research protocol: anatomical feedback constraints in PC-ALM

Status: prospective diagnostic protocol, 2026-09-19. Confirmatory training NOT authorized by this protocol until gates pass. This document is hashed before the diagnostic run. No claim of proven novelty or predicted task accuracy.

## Topic and scope
**실제 MaleCNS 회로에서 PC-ALM의 학습 신호 전달 제약과 보정 가능성.**
Distinguish forward anatomy, inference/error communication, and optimization. The target is whether PC-ALM works and improves learning on a measured circuit, not whether fly topology beats artificial topology.

RQ1: Does exact-feedback ALM recover the correct constrained-network gradient on a frozen linear recurrent MaleCNS subgraph?
RQ2: When auxiliary error signals may travel only on measured directed edges, does stable inference still produce biased credit?
RQ3: Can routing or learned feedback close that gap under equal communication/computation budgets in nonlinear learning?
RQ4: Does any improvement persist on held-out real stimuli and independently selected circuits?

RQ1 is a correctness control, RQ2 a diagnostic, RQ3–4 prospective research. Weight transport, feedback alignment, arbitrary-graph PC, ALM stability, and temporal PC already exist. Neither an ALM application nor the matrix below is claimed new. Candidate contribution: a measured anatomy-to-credit feasibility/bias audit plus a validated budget-constrained remedy. Novelty gate remains OPEN: targeted feedback/PC literature and code comparison must be completed before claiming a new method.

## References and scope of prior art
1. Seely & Gould (2026), Augmented Lagrangian Predictive Coding, https://arxiv.org/html/2605.31022v1 Appendix C. Linear frozen-weight stability and exact credit; directly inherited here.
2. Official implementation https://github.com/SakanaAI/pc-alm commit 660747f61a8a7e547c0ecd2c48c8883380a7d1f6. Hidden constraints, output loss separate, per-sample inference, primal-then-dual, pre-dual weight timing. This pilot is an explicit recurrent linear adaptation, NOT a reproduction of its residual MLP benchmark.
3. Millidge, Tschantz & Buckley, Predictive Coding Approximates Backprop along Arbitrary Computation Graphs, https://arxiv.org/abs/2006.04182 . Arbitrary computation graphs are prior art, not equivalent to unqualified cyclic fixed-point inference.
4. Millidge et al. (2024), Predictive coding networks for temporal prediction, https://doi.org/10.1371/journal.pcbi.1011183 . Temporal PC is prior art.
5. Wang, Zhang & Chen (2025), An Augmented Lagrangian Method for Training Recurrent Neural Networks, https://doi.org/10.1137/23M1627614 . ALM RNN training is prior art; publisher abstract inspected, full proof not audited.
6. Liao et al. (2016), How Important Is Weight Symmetry in Backpropagation?, https://ojs.aaai.org/index.php/AAAI/article/view/10279 . Asymmetric feedback is prior art.
7. Lappalainen et al. (2024), Connectome-constrained networks predict neural activity across the fly visual system, https://doi.org/10.1038/s41586-024-07939-3 ; https://github.com/TuragaLab/flyvis . Functional connectome model reference, not MaleCNS.
8. MaleCNS https://male-cns.janelia.org/download/ . v1.0 measured anatomy; local raw file hashes in ../credit_rebuild/manifest.json.
9. Sakana discussion https://pub.sakana.ai/pc-alm/ . Dual leak and temporal tasks already discussed.

## Data and explicit assumptions
Use existing 132-neuron, 2663-edge thresholded subset (at least 5 measured synapses per edge), all selected edges retained. Verify selected edge counts and weights against the local official connectivity feather before inference. This subset was selected in prior exploratory work, not a fresh independent sample. It is not the whole brain. The 22 unknown transmitter signs receive seeded engineering signs; inferred known signs are modeling assumptions, not guarantees for every receptor.
Let S[i,j]=1 mean j -> i exists; W[i,j] is signed log(1+synapse_count), with source-neuron sign, times seeded positive factors and normalized to spectral norm g in {0.4,0.8}. Magnitudes and dynamics are engineered, not measured physiology. Input u is a synthetic random current on all nodes; readout C selects the existing 24 descending neurons. Targets are seeded random vectors, only to exercise nonzero credit. No sensory/generalization claim.

## Definitions and derivation
Column states, h = W h + u, A=I-W. Because ||W||_2=g<1, A is invertible. Forward equilibrium h*=A^{-1}u. Loss ell=0.5||Ch-y||^2. Residual r=Ah-u.
L_rho=ell+lambda^T r+(rho/2)||r||^2; rho=1.
Exact stationary multiplier lambda*=-A^{-T}C^T(Ch*-y).
True masked parameter gradient G*=-lambda* h*^T elementwise S. This is implicit differentiation, NOT finite-horizon BPTT. Sign-boundary projection is outside this frozen-weight diagnostic.

Exact feedback F=A^T. Anatomy-restricted surrogate F=I-(W^T elementwise S). Identity is local self-computation, not an anatomical synapse. The latter is NOT an exact gradient of L_rho and is explicitly called masked-feedback ALM surrogate. Feedback weight signs are not constrained to transmitter signs: this is only a directed-support constraint, not a biologically complete model. Extra reverse channels in exact ALM are computational assumptions.

Updates: h+ = h-eta[C^T(Ch-y)+F(lambda+rho r)]; lambda+=lambda+alpha(Ah+-u).
Weight credit at a budget endpoint uses h+ and the pre-update lambda, matching the pre-dual convention. PC: exact F, alpha=0; exact ALM and masked surrogate: alpha=0.2. eta=0.2 fixed, no outcome tuning.
Set B=C^TC, K=I-eta(B+rho FA). Joint error matrix M=[[K,-eta F],[alpha A K,I-alpha eta AF]]. Before running, compute spr(M), exact fixed point, and finite-budget error prediction by M^t. spr(M)<1 predicts asymptotic stability for ALM. Non-normal transient growth is possible; spr(M) alone does not bound finite-budget error. PC uses K, not the joint M with its unused dual eigenvalues.
For invertible F, a stable masked surrogate reaches r=0 but lambda_F=-F^{-1}C^T(Ch*-y). Thus feasibility does NOT imply correct credit. Predicted relative bias is ||(-lambda_F h*^T-G*) elementwise S||_F / ||G*||_F, computed BEFORE inference. This is an algebraic diagnostic, not a novel convergence theorem.

## Frozen diagnostic design
5 seeds 42000–42004 x 2 gains x 3 methods = 30 trajectories. Budgets 8,32,128,512,4096; 4096 is an equilibrium diagnostic, NOT a competitive practical budget. Start h=h*, lambda=0, identical W,u,y for paired methods. No training, validation, test accuracy, or population significance claim.
Primary: masked gradient relative error. Secondary: cosine, residual, normalized state/dual distance to predicted fixed point, predicted-vs-observed recurrence error. Save predicted stability and bias before any trajectory runs.
Correctness gates: directional finite-difference gradient relative discrepancy <1e-5; analytic recurrence matches independently updated state <1e-9 relative; raw selected edge weights and counts agree; exact stable ALM reaches relative gradient error <1e-5 by 4096. Failure stops progression and is reported; do not silently tune or replace seeds.

## Expected findings BEFORE run
Exact ALM gradient bias at equilibrium is zero by algebra, conditional on stability. Masked surrogate bias is generically nonzero; magnitude must be computed from the instantiated circuit, not guessed. PC's finite-penalty solution need not equal forward equilibrium. No prediction that ALM beats BP in task accuracy. No numerical accuracy-improvement percentage is justified by the current evidence.

## Main-study gates and future design
G0 correctness/diagnostic gates pass. G1 nearest-prior-art audit explicitly distinguishes proposed remedy from feedback alignment, arbitrary-graph PC and existing ALM preconditioners; no uniqueness claim from failed web search. G2 reproduce official PC-ALM baseline and a functional flyvis baseline with pinned versions. G3 nonlinear pilot learns above a simple baseline and shows stable gradients with anatomically justified inputs. Only then freeze confirmatory protocol.
Main candidate comparisons: exact BP/BPTT; standard PC; exact-feedback ALM; support-constrained surrogate; one justified correction. Equal task, forward graph, trainable weights, initialization and optimizer; per-method equal tuning trials on development data. Feedback routing cost, inference steps, messages, wall time, peak memory and any learned feedback parameters counted. Separate oracle controls.
Use 5 development seeds; reserve 20 new paired confirmation seeds; three independently specified circuit selections. If real movie data used, split by source movie/recording before crops/windows. Real anatomy and rendered/synthetic stimuli must be labeled separately. Primary held-out task loss; mechanistic secondary gradient error, feasibility, failure rate. Report paired effect sizes/95% intervals and Holm adjustment for predeclared primary comparisons; seeds are not independent biological animals. Freeze noninferiority margin and power/precision plan using development variance before confirmation. Never select checkpoints or tune on test outcomes.
A promising remedy must reduce development credit error at matched communication budget and avoid task degradation; a 20% error reduction can be a practical go/no-go threshold, explicitly an engineering target rather than predicted outcome. No SCI acceptance guarantee. If novelty or task gates fail, publish an honest diagnostic rather than manufacture a positive claim.
