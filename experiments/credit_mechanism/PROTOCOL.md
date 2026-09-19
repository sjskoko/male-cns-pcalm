# Direction is not descent: finite-budget PC-ALM on sign-constrained connectomes

Status: exploratory protocol written before this pilot. Not a registered confirmatory study.

## Question and candidate contribution

Does finite-budget PC-ALM lose useful credit between raw gradient alignment and the actual optimizer/sign-projected weight step? The target is learning on fixed MaleCNS wiring, not topology efficiency.

Candidate contribution: a mechanism-resolving evaluation separating gradient direction, layerwise magnitude, optimizer state and sign projection. Combining these interventions with connectome constraints may be useful, but priority/novelty is unverified. A damped-dual variant is a testable engineering hypothesis, not an established novel algorithm.

## Intervention arms

1. BP: reference training.
2. PC: alpha=0.
3. ALM: alpha=.2.
4. ALM direction + BP layer norm: g_l = g_ALM,l * ||g_BP,l|| / ||g_ALM,l||.
5. BP direction + ALM layer norm: reverse intervention.
6. Damped ALM: lambda <- .9 lambda + .2 r, all other updates unchanged.

Apply sparse masks before norm interventions; retain zero directions as zero. Both oracle arms explicitly use BP and cannot be marketed as local-learning algorithms. They are interventions at the current model state, not comparisons of gradients from different trained models. Training trajectories subsequently diverge, so aggregate differences alone do not prove mediation.

Damping changes the fixed point: stationarity gives lambda=alpha*r/(1-beta), so r need not vanish. No exact-BP convergence claim is inherited from standard ALM. beta=1 must reproduce the baseline's numerical update.

## Fixed design

Use the existing 132-neuron/1,001-edge circuit and synthetic nonlinear/linear teachers. 62.4% of selected edges remain excluded. Known signs are projected; ambiguous signs remain free.

Pilot: 2 tasks × 2 optimizers × 2 seeds (2100–2101) × 6 methods = 48 fits; 10 epochs, T=8.

Full exploratory profile: 2 tasks × 2 optimizers × 20 fresh seeds (2200–2219) × 3 budgets (4,8,32) × 6 methods = 1,440 fits; 60 epochs. This profile is implemented but is NOT the final publication experiment.

Common optimizer LR=.003; activity rate=.05; ALM dual rate=.2; batch=64. Same data, initialization and shuffle for paired methods. Adam uses .9/.999 and epsilon=1e-8. SGD has no momentum. Fixed identical LRs are mechanistic controls, not evidence that each optimizer is optimally tuned.

Metrics: final validation/test MSE, hidden raw-gradient cosine, gradient norm ratio, actual projected-step cosine to masked BP, dot(g_BP, W_old-W_new), projection clipping fraction and measured same-minibatch loss change. Diagnostic sampling is the last minibatch of each epoch, not all updates or independent validation observations. Dot products correspond to half-sum squared loss; reported MSE has different scaling but the same descent sign.

Every method currently computes BP for diagnostics, including local methods. Runtime therefore includes oracle instrumentation and is NOT an efficiency comparison. Same inference steps do not imply equal compute. Nonfinite runs must be recorded as failures, not silently excluded from success claims. New output directories are required; published evidence is not overwritten.

## Falsifiable hypotheses

- H1: ALM's raw alignment advantage need not survive optimizer/projection. Test the within-run gap across epochs; absent gaps weaken this explanation.
- H2: Norm intervention improves ALM more than direction intervention if scale is the main bottleneck. If not, reject a scale-only explanation.
- H3: Damping reduces harmful steps and test error versus ALM. No improvement or a stability/performance tradeoff counts against the proposed remedy.

Pilot: descriptive statistics only, no significance claims from two seeds.

## What is still required for a journal submission

Freeze a separate confirmatory protocol after development: equal tuning opportunities on fresh development seeds, learning-rate/dual/beta sweeps, at least 20 held-out paired seeds (power justified from development variance), signed versus unconstrained controls, multiple independently selected real circuits and at least one external task. Current runner does NOT implement these extensions.

Plan contrasts ALM–PC, ALM–damped, ALM–norm-oracle, ALM–direction-oracle in every predeclared stratum; two-sided paired tests with Holm across the entire family, paired bootstrap confidence intervals and all failures. Do not treat epochs as independent replications. Report compute without BP diagnostic overhead separately. Match wall-time in an additional experiment. Include beta=1/alpha=0 equivalence, numerical gradient and projection checks.

Paper structure: motivating discrepancy → constrained descent analysis → intervention study → local remedy → generalization and limits. A useful theoretical target is the smoothness bound L(W-d) <= L(W)-<grad L,d>+(K/2)||d||², with d the actual feasible update. This standard bound is motivation, not a new theorem or a proof of convergence for damping.

## Prior work and claim boundaries

- Seely & Gould, [Augmented Lagrangian Predictive Coding](https://arxiv.org/abs/2605.31022): baseline ALM and credit propagation already exist.
- Ofner et al., [Predictive coding, precision and natural gradients](https://arxiv.org/abs/2111.06942): precision and optimization geometry already studied.
- Salvatori et al., [Learning on Arbitrary Graph Topologies via Predictive Coding](https://arxiv.org/abs/2201.13180): arbitrary-graph PC already exists.

These sources were checked at abstract level for scope; this is not an exhaustive full-text novelty review. Do not claim first use of damping, norm interventions, arbitrary graphs or connectome-inspired learning. The pilot cannot establish SCI/SCIE publication readiness.
