import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import study

HERE=Path(__file__).resolve().parent
out=HERE/'budget_audit_results';out.mkdir(exist_ok=False);(out/'runs').mkdir()
study.dump(out/'manifest_before_audit.json',{'hashes':{p.name:study.digest(p) for p in [Path(__file__),HERE/'BUDGET_AUDIT_PROTOCOL.md',HERE/'study.py']}})
sp=json.loads((HERE/'results/split.json').read_text());split=[np.array(sp[k]) for k in ['train','validation','test']]
d=load_digits();x=(d.data/16).T;y=np.eye(10)[d.target].T
selected=json.loads((HERE/'results/selected_before_evaluation.json').read_text())
original_project=study.project
snapshot={}
def capture(w,c,mask,signs,output):
    w,c=original_project(w,c,mask,signs,output)
    snapshot['w']=w.copy();snapshot['c']=c.copy()
    return w,c
study.project=capture
rows=[];replay_checks=[]
for seed in range(46000,46003):
    row=study.fit(seed,'alm',selected['alm'],split,x,y,'development_replay',out)
    old=json.loads((HERE/'results/runs'/f"development_{seed}_alm_{selected['alm']}.json").read_text())
    err=abs(row['validation_loss']-old['validation_loss']);assert err<1e-12
    replay_checks.append({'seed':seed,'validation_replay_error':err})
    w=snapshot['w'];c=snapshot['c'];_,_,b,mask,_,_=study.initialize(seed)
    ix=split[0][:32];u=b@x[:,ix];yy=y[:,ix]
    gref=study.gradients(w,c,u,yy,'implicit')[0]*mask
    a=np.eye(len(w))-w
    for method in ['pc','alm','masked']:
        f=np.eye(len(w))-w.T*mask if method=='masked' else a.T
        k=np.eye(len(w))-.1*(c.T@c+f@a)
        m=k if method=='pc' else np.block([[k,-.1*f],[.2*a@k,np.eye(len(w))-.02*a@f]])
        radius=float(max(abs(np.linalg.eigvals(m))))
        for t in [8,32,128,512]:
            gw,_,res=study.gradients(w,c,u,yy,method,steps=t);g=gw*mask
            rows.append({'seed':seed,'method':method,'steps':t,'spectral_radius':radius,
                'gradient_error':float(np.linalg.norm(g-gref)/np.linalg.norm(gref)),
                'gradient_cosine':float(np.sum(g*gref)/(np.linalg.norm(g)*np.linalg.norm(gref))),
                'residual':res})
pd.DataFrame(rows).to_csv(out/'probes.csv',index=False)
pd.DataFrame(rows).groupby(['method','steps'])[['gradient_error','gradient_cosine','residual']].mean().to_csv(out/'summary.csv')
study.dump(out/'replay_checks.json',replay_checks)
print(pd.read_csv(out/'summary.csv').to_string(index=False),flush=True)
