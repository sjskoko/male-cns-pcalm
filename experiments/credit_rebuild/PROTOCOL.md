# Fresh PC-ALM credit-assignment study

This is a new execution, not recovery of the lost 994256b experiment. Do not
reuse that experiment's numerical claims as evidence for this run.

Question: on a fixed real MaleCNS sparse graph, does PC-ALM improve learning
and gradient alignment relative to PC? Topology efficiency is not tested.

Use the repository's frozen visual_projection -> cb_intrinsic -> cb_intrinsic
-> descending_neuron assignment (132 neurons). Reload official raw weights,
verify SHA-256 and retain adjacent forward edges with >=5 synapses. Report
discarded recurrent/skip edges. Known NT signs follow the conservative original
assignment; ambiguous signs remain unconstrained. This is a feedforward
projection, not a simulation of the full recurrent CNS.

Independent implementation: NumPy float64 tanh hidden layers, linear output,
no bias. PC-ALM follows Seely & Gould Algorithm 1: T-1 simultaneous primal / dual
cycles, then a final primal update, then local weight gradients. BP is only a
baseline and diagnostic. Exact scaled constraints z=r/s define TP-PC-ALM;
s=clip((in_degree/median_degree)^0.25, .75, 1.5). A shuffled-scale control
retains the scale distribution but permutes it within each layer.

Tasks: realizable nonlinear teacher regression (same graph), plus an independent
fixed linear teacher regression (32 inputs, 24 outputs). Both are synthetic;
neither is a measured fly behavior. Data: 512 train / 256 validation / 256 test,
Gaussian inputs, 1% target noise on train only. Separate RNG streams for teacher,
data, initialization and minibatch order. All methods are paired within seeds.

Methods: BP, PC, PC-ALM, TP-PC-ALM, shuffled TP-PC-ALM. Adam, batch 64,
60 epochs, T=8. Development seeds 1100-1104; final seeds 1200-1219.
Equal 8-candidate search budget per method using validation MSE only:
lr in {.001,.003}, activity rate in {.05,.1}, dual rate in {.2,.5} for ALM.
For PC use lr {.0005,.001,.003,.006} x activity {.05,.1}; for BP use
8 learning rates {.0001,.0003,.0005,.001,.002,.003,.006,.01}.
Choose per task/method by mean development validation MSE. Freeze before final
data are evaluated. Never tune on final test results.

Primary contrasts: PC vs PC-ALM, PC-ALM vs TP-PC-ALM, shuffled vs TP-PC-ALM,
on each task (6 tests). Two-sided paired Wilcoxon and Holm across all six;
paired bootstrap mean differences with 95% CI (20,000 resamples), wins and
seed-level values. CI are pointwise, not multiplicity-adjusted. Report all
outcomes including failure and negative findings; no success-conditioned reruns.

Diagnostics: initial BP-gradient cosine at T=1,2,4,8,16,32, separately for hidden
weights (primary alignment) and all weights; residual, runtime, clean test MSE,
noisy-input MSE. CPU runtime is descriptive, not a hardware efficiency claim.

Candidate novelty: degree-scaled local AL constraints on a connectome-derived
network. This is a hypothesis, not a verified first-in-literature claim. A paper
requires multiple pathways, recurrent extension, wider hyperparameter search,
and independent measured targets before biological claims.

Sources: https://arxiv.org/abs/2605.31022 ; https://male-cns.janelia.org/download/
