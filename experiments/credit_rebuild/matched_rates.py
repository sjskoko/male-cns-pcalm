"""Execute the frozen secondary control; reuses audited local updates."""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from study import HERE, TASKS, circuit, digest, dump, fit

assert json.loads((HERE/'frozen.json').read_text())['code_sha256']==digest(HERE/'study.py')
masks,signs=circuit()
rows=[]
for task in TASKS:
    for lr in [.003,.006]:
        for seed in range(1300,1320):
            for method in ['pc','pcalm']:
                path=HERE/f'checkpoints/matched_{task}_{lr}_{method}_{seed}.json'
                if path.exists(): row=json.loads(path.read_text())
                else:
                    row=fit(masks,signs,task,seed,method,(lr,.05,0 if method=='pc' else .2),final=True)[0]
                    dump(path,row)
                rows.append(row)
        print('completed matched',task,lr,flush=True)
frame=pd.DataFrame(rows);frame.to_csv(HERE/'matched_trials.csv',index=False)
stats=[];rng=np.random.default_rng(651)
for task in TASKS:
    for lr in [.003,.006]:
        pivot=frame[(frame.task==task)&(frame.lr==lr)].pivot(index='seed',columns='method',values='test_mse')
        d=(pivot.pc-pivot.pcalm).to_numpy();boot=rng.choice(d,(20000,20)).mean(1)
        stats.append({'task':task,'lr':lr,'pc_mse':float(pivot.pc.mean()),'pcalm_mse':float(pivot.pcalm.mean()),'delta':float(d.mean()),'reduction_percent':float(d.mean()/pivot.pc.mean()*100),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'wins':int((d>0).sum()),'p':float(wilcoxon(d).pvalue) if np.any(d) else 1.})
last=0.
for rank,i in enumerate(np.argsort([r['p'] for r in stats])):
    last=max(last,min(1.,(4-rank)*stats[i]['p']));stats[i]['holm_p']=last
dump(HERE/'matched_comparisons.json',stats)
(HERE/'MATCHED_RESULTS.md').write_text('# Secondary matched-rate results\n\n'+pd.DataFrame(stats).to_markdown(index=False)+'\n\nPositive delta favors PC-ALM. Separate secondary family; see MATCHED_RATE_PROTOCOL.md.\n')
print(pd.DataFrame(stats).to_string(index=False),flush=True)
