# Temporal MaleCNS study: exploratory protocol

Written before execution. Research question: can finite-inference PC-ALM train a signed recurrent network using measured MaleCNS wiring on a temporal motion task, and improve over PC within the same graph?

## Actual versus generated data

Actual: official MaleCNS v1.0-derived assignments and all 2,663 thresholded edges among 132 selected neurons, previously extracted and checksummed in credit_rebuild. This is a visual-projection/central-brain/descending subset, not a complete optic-lobe circuit. The full graph restores intra-layer, backward and skip edges. Edges to unselected neurons remain absent. Refer to ../credit_rebuild/manifest.json for raw-source hashes and https://male-cns.janelia.org/download/ for attribution/data terms.

Actual images: sklearn load_digits, 1,797 measured 8×8 handwritten digit images; source and description: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html . No raw images are redistributed here. The manifest hashes the loaded images and records sklearn version.

Generated: six-frame periodic translations of those images, with four balanced velocities (+/-1 horizontal or vertical pixel/frame). Starting offsets are random so absolute position is not a direction cue. Frames are downsampled to 4×8. This is NOT natural video, measured optic flow, neural activity or fly behavior. Wraparound and low resolution are limitations.

Split original image identities before generating sequences: fixed permutation seed 731, first 1000 train, next 400 validation, remainder test. Each run samples 128/64/128 distinct source images from these disjoint pools. This is image-disjoint, not guaranteed writer-disjoint. No neuron-to-pixel retinotopy is claimed: a fixed sorted-ID adapter injects 32 features into 32 visual projection neurons. Outputs use the 24 descending neurons via a trainable unconstrained 2D linear readout.

## Dynamics and learning

h_0=0; f_t=.5 h_(t-1)+.5 tanh(h_(t-1) W+x_t B); h_t=f_t.

Every anatomical edge has one model time-step delay. Leak .5 is an engineering assumption, not a measured membrane constant. Full and forward-only graphs use identical dynamics; even forward-only has leak memory. W is masked, known presynaptic signs projected; unknown signs remain unconstrained. Initial |weights| depend on log(1+synapse count) and are scaled to spectral norm .8. This bound is not enforced during training. Recurrent spectral gain normalization means shared edges have different normalized magnitudes across graph conditions; graph comparisons cannot isolate recurrence alone.

Time unrolling turns recurrence into local constraints r_t=h_t-f_t. Objective: .5||h_T C-y||² + sum_t(lambda_t*r_t+.5||r_t||²). All temporal states undergo simultaneous primal updates; duals follow updated residuals. T_infer-1 primal/dual rounds plus one final primal round. Shared W gradients sum contributions from all time edges. The method is a temporal adaptation of ALM, not an unmodified reproduction of the Sakana model. Temporal neighbor access, a stored trajectory and weight transposes are required; this does NOT solve online temporal credit assignment or biological weight transport.

Reference BP uses exact BPTT through the same six steps. PC uses alpha=0; ALM alpha=.2. Both use 8 inference steps and primal rate .05. Adam LR=.003, betas .9/.999, epsilon 1e-8; 20 epochs; four batches of 32. Final epoch is evaluated without test-set model selection. No tuning on these seeds. This is a fixed-setting feasibility study, not an optimally tuned comparison.

## Experiment and outcomes

10 paired seeds 3100–3109 × 2 graphs × 3 learners = 60 fits. Same graph methods share exact initialization, data and batch order. Full graph 2,663 edges; forward-only graph 1,001 edges: unequal parameter counts, so graph differences are descriptive and not proof of superior biological topology.

Primary: within-graph paired PC vs ALM final test MSE, two-sided Wilcoxon, Holm across two comparisons, 10,000 paired bootstrap resamples (pointwise 95% intervals). These are exploratory tests, not preregistered confirmatory evidence. Report failures and paired sample counts.

Secondary: nearest-direction accuracy (chance 25%), middle-two-frame occlusion MSE and randomly shuffled-time MSE. Zero-vector predictor MSE=.5 because labels are cardinal unit vectors. Shuffling is a stress test, not an isolated test of memory; it changes motion statistics. Training validation loss and constraint residual are saved each epoch. Test perturbations are not used for selection. Runtime is measured per fit, not energy efficiency.

## Required next stage

Real video/flow data, multiple biologically grounded circuits and retinotopic adapters, matched-capacity controls, trained time constants, more development/held-out seeds and equal hyperparameter budgets remain unimplemented. Compare restoration levels and matched-edge controls before attributing gains to recurrence. Any full-graph failure may reflect short observation windows, sparse path delays, feature adapters or poor hyperparameters rather than useless biological wiring.

References: PC-ALM https://arxiv.org/abs/2605.31022 ; arbitrary-graph PC https://arxiv.org/abs/2201.13180 ; connectome-constrained visual modeling https://www.nature.com/articles/s41586-024-07939-3 . This study does not claim novel recurrent PC by itself.
