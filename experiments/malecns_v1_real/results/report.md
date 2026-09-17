# Visual–motor topology × learning-rule benchmark

- Base graph: `MaleCNS projection`
- Device: `cpu`
- Layer sizes: `[32, 40, 36, 24]`
- Base edges: `1001`
- Input layout: `connectome_metadata`

> This run uses a MaleCNS topology with a synthetic proxy task. It is not evidence of fly behavior.

| Topology | Method | Runs | Test accuracy | Accuracy AUC | BP-gradient cosine | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| native | bp | 10 | 0.914 ± 0.141 | 0.619 | 1.000 | 0.38 |
| native | pc | 10 | 0.941 ± 0.080 | 0.622 | 0.585 | 2.41 |
| native | pcalm | 10 | 0.936 ± 0.091 | 0.619 | 0.601 | 3.14 |
| degree_preserving | bp | 10 | 0.923 ± 0.151 | 0.638 | 1.000 | 0.39 |
| degree_preserving | pc | 10 | 0.943 ± 0.120 | 0.643 | 0.592 | 2.42 |
| degree_preserving | pcalm | 10 | 0.941 ± 0.121 | 0.641 | 0.612 | 3.16 |
| random | bp | 10 | 0.973 ± 0.057 | 0.658 | 1.000 | 0.42 |
| random | pc | 10 | 0.969 ± 0.065 | 0.663 | 0.637 | 2.51 |
| random | pcalm | 10 | 0.968 ± 0.067 | 0.661 | 0.649 | 3.14 |

## Paired native-minus-control effects

Positive values favor the native topology. Runs are paired by seed.

| Method | Control | Runs | Accuracy delta | Native win rate | AUC delta | Gradient-cosine delta |
|---|---|---:|---:|---:|---:|---:|
| bp | degree_preserving | 10 | -0.009 ± 0.110 | 0.20 | -0.019 | +0.000 |
| bp | random | 10 | -0.059 ± 0.106 | 0.00 | -0.040 | +0.000 |
| pc | degree_preserving | 10 | -0.002 ± 0.116 | 0.20 | -0.021 | -0.006 |
| pc | random | 10 | -0.028 ± 0.060 | 0.20 | -0.040 | -0.052 |
| pcalm | degree_preserving | 10 | -0.005 ± 0.112 | 0.20 | -0.021 | -0.011 |
| pcalm | random | 10 | -0.032 ± 0.065 | 0.20 | -0.041 | -0.048 |

Interpret accuracy differences only after running multiple seeds. The primary MaleCNS test is
native versus degree-preserving rewiring under PC-ALM; the virtual graph only validates the workflow.
