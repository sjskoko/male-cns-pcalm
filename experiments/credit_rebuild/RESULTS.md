# Fresh experiment results

200 final fits; 2 tasks × 5 methods × 20 paired seeds. Development: 400 fits.

| task      | method   |        mse |         sd |       r2 |   initial_mse |   seconds |   noisy_mse |
|:----------|:---------|-----------:|-----------:|---------:|--------------:|----------:|------------:|
| linear    | bp       | 0.378105   | 0.0266027  | 0.625356 |      1.3199   |  0.129108 |   0.404269  |
| linear    | pc       | 0.558684   | 0.0523253  | 0.446139 |      1.3199   |  0.877991 |   0.584349  |
| linear    | pcalm    | 0.557552   | 0.0432489  | 0.44757  |      1.3199   |  0.886335 |   0.580731  |
| linear    | shuffle  | 0.560713   | 0.0459912  | 0.444415 |      1.3199   |  0.895223 |   0.583833  |
| linear    | tp       | 0.557858   | 0.0429997  | 0.447285 |      1.3199   |  0.899488 |   0.580884  |
| nonlinear | bp       | 0.00582123 | 0.00191858 | 0.980819 |      0.431831 |  0.13124  |   0.020558  |
| nonlinear | pc       | 0.0109094  | 0.00291229 | 0.96354  |      0.431831 |  0.864704 |   0.0248884 |
| nonlinear | pcalm    | 0.0172144  | 0.00339859 | 0.942526 |      0.431831 |  0.864317 |   0.0307642 |
| nonlinear | shuffle  | 0.0172888  | 0.0033963  | 0.942238 |      0.431831 |  0.86231  |   0.0308554 |
| nonlinear | tp       | 0.0172289  | 0.00338993 | 0.94245  |      0.431831 |  0.880627 |   0.0307679 |

## Paired comparisons

| task      | left    | right   |    delta_mse |       ci_low |      ci_high |           p |   wins |   reduction_percent |      holm_p |
|:----------|:--------|:--------|-------------:|-------------:|-------------:|------------:|-------:|--------------------:|------------:|
| nonlinear | pc      | pcalm   | -0.00630503  | -0.00705768  | -0.0055884   | 1.90735e-06 |      0 |         -57.7945    | 1.14441e-05 |
| nonlinear | pcalm   | tp      | -1.45201e-05 | -0.000129527 |  0.000100444 | 0.812355    |     10 |          -0.0843487 | 1           |
| nonlinear | shuffle | tp      |  5.98269e-05 | -8.09416e-05 |  0.000203249 | 0.570597    |     11 |           0.346045  | 1           |
| linear    | pc      | pcalm   |  0.00113139  | -0.00955419  |  0.0111746   | 0.674223    |     10 |           0.20251   | 1           |
| linear    | pcalm   | tp      | -0.000305752 | -0.00322909  |  0.00228316  | 0.869488    |     12 |          -0.0548383 | 1           |
| linear    | shuffle | tp      |  0.0028555   | -0.00184533  |  0.00701448  | 0.113987    |     14 |           0.509262  | 0.569935    |

## Scope

Real MaleCNS feedforward projection; synthetic targets. No whole-CNS, behavior, topology-efficiency or publication-readiness claim. Positive delta favors right. Two-sided Wilcoxon; Holm over six comparisons. CI are pointwise paired bootstrap intervals.

Frozen settings: frozen.json. Full seed-level evidence: trials.csv; alignment.csv; curves.csv.
