# Contributing

Independent reproductions and critical feedback are welcome. English and Korean issues are both fine.

## Start small

1. Run the CPU verification commands in [README.md](README.md).
2. Choose a scoped item from [ROADMAP.md](ROADMAP.md).
3. Open an issue describing the intended change before a large implementation.
4. Submit a pull request with commands, tests, and a clear explanation.

## Preserve the evidence

Do not edit the frozen `experiments/credit_rebuild/study.py`, protocols, settings, or saved results to improve a reported number. Create a new experiment directory for changed algorithms or settings. Preserve source-data checksums and record all seeds, including failed runs.

Tune only on development/validation data; reserve fresh evaluation seeds. Compare paired initialization, data, optimizer budgets, and inference budgets. Report uncertainty, runtime, and negative results. Separate initial gradient alignment from final task performance.

## Verification

```bash
python -m pytest -q experiments/credit_rebuild/test_study.py
python experiments/credit_rebuild/verify_results.py
```

For changes to the original PyTorch package, also run the root tests and lint using the development environment described in README.ko.md. Never claim a test passed without running it.

## Reproduction reports

Include commit SHA, OS, Python/dependency versions, exact command, task/method/seeds, expected versus observed result, and a minimal sanitized log. Do not upload credentials, private datasets, or unrelated personal information.

Code contributions use the repository MIT license. Preserve upstream data attribution and applicable data terms. Be respectful; disagreement about methods and results is welcome.
