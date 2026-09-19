# Research roadmap

These are proposed contributions, not completed features or promised performance gains.

| Scope | Contribution | Acceptance criteria |
|---|---|---|
| Small | Independent CPU reproduction | Exact commit, environment, test output and aggregate audit; no changes to frozen records |
| Small | Read-only tutorial notebook | Loads saved results, explains synthetic targets and graph exclusions, runs without GPU or overwriting files |
| Medium | Finite-budget stability study | Fresh development/evaluation seeds; sweep inference budget, primal/dual rates and optimizer rate; report all attempted settings |
| Medium | Alignment over training | Measure hidden-weight alignment and magnitude at fixed checkpoints; compare final error without selecting on test outcomes |
| Large | Additional connectome circuits | Predeclare selection rules, report excluded edges and signs, preserve paired controls |
| Large | Recurrent-graph formulation | Explicit objective/update derivation, numerical checks, stability criteria and controlled baselines |

Priority question: why does improved initial gradient alignment fail to produce better final performance under the current finite-budget settings?

A contribution is valuable if it narrows uncertainty—even if it does not improve accuracy.
