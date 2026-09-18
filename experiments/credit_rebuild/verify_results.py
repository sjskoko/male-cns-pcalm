"""Audit sample counts, pairing, frozen configuration and serialized aggregates."""
import json

import numpy as np
import pandas as pd

from study import HERE, METHODS, TASKS, digest, dump

frozen=json.loads((HERE/'frozen.json').read_text())
assert digest(HERE/'study.py')==frozen['code_sha256']
assert digest(HERE/'PROTOCOL.md')==frozen['protocol_sha256']
assert digest(HERE/'circuit.npz')==frozen['circuit_sha256']
dev=pd.read_csv(HERE/'development.csv');trials=pd.read_csv(HERE/'trials.csv')
assert len(dev)==400 and len(trials)==200
assert set(dev.seed)==set(range(1100,1105))
assert set(trials.seed)==set(range(1200,1220))
assert not trials.duplicated(['task','method','seed']).any()
assert np.isfinite(trials.select_dtypes('number')).all().all()
for task in TASKS:
    for method in METHODS:
        r=trials[(trials.task==task)&(trials.method==method)]
        assert len(r)==20
        np.testing.assert_allclose(r[['lr','eta','alpha']].to_numpy(),np.tile(frozen['selected'][task][method],(20,1)))
    assert (trials[trials.task==task].groupby('seed').initial_test_mse.nunique()==1).all()
summary=pd.read_csv(HERE/'summary.csv').set_index(['task','method'])
np.testing.assert_allclose(trials.groupby(['task','method']).test_mse.mean(),summary.mse,rtol=1e-12)
matched=pd.read_csv(HERE/'matched_trials.csv')
assert len(matched)==160 and set(matched.seed)==set(range(1300,1320))
assert not matched.duplicated(['task','method','seed','lr']).any()
assert np.isfinite(matched.select_dtypes('number')).all().all()
result={'status':'passed','development_fits':400,'primary_fits':200,'secondary_fits':160,'total_fits':760,'all_primary_and_secondary_metrics_finite':True,'frozen_code_and_protocol_match':True,'paired_initialization_match':True}
dump(HERE/'verification.json',result)
print(result)
