# Secondary matched-rate control

Added after development tuning, while primary final runs were ongoing, before
inspecting any primary final metric. Development selected PC lr=.006 and ALM
lr=.003, both at their grid maxima. Therefore primary between-method differences
cannot isolate the effect of multipliers from learning-rate choice.

Secondary fixed experiment: fresh seeds 1300-1319, both tasks; PC and PC-ALM
at each shared lr=.003 and .006, shared activity rate=.05, T=8, 60 epochs.
Alpha is 0 for PC and .2 for PC-ALM. All other choices match study.py.
Total: 2 tasks × 2 rates × 2 methods × 20 seeds = 160 fits. No tuning or
selection using these results. Report all four paired contrasts, two-sided
Wilcoxon and Holm within this separate secondary family, pointwise 95% paired
bootstrap intervals. This family does not replace the six primary contrasts.

Motivation is a concrete confound identified from DEVELOPMENT results, not
attempting to reverse a negative final result. Any combined success claim must
disclose both experiment families and finite-grid limitations.
