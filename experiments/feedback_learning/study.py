"""Development-stage equilibrium learning; frozen split, paired seeds, no test tuning."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import t as student_t

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHODS=['implicit','pc','alm','masked','readout']

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))

def graph():
    nodes=pd.read_csv(ROOT/'experiments/malecns_v1_real/assignments.csv').sort_values('body_id')
    edges=pd.read_csv(ROOT/'experiments/credit_rebuild/all_selected_edges.csv')
    ids=nodes.body_id.to_numpy(); n=len(ids)
    syn=np.zeros((n,n))
    syn[np.searchsorted(ids,edges.post),np.searchsorted(ids,edges.pre)]=edges.synapses
    return syn>0,nodes.sign.to_numpy(),nodes.layer.to_numpy()==3,syn

def initialize(seed):
    mask,signs,out,syn=graph();rng=np.random.default_rng(seed);n=len(mask)
    w=rng.normal(size=(n,n))*np.log1p(syn)
    w=np.where(signs[None,:]!=0,np.abs(w)*signs[None,:],w)*mask
    w*=.4/np.linalg.norm(w,2)
    c=rng.normal(size=(10,n))*.05*out
    b=rng.normal(size=(n,64))/8
    return w,c,b,mask,signs,out

def forward(w,c,u):
    h=np.linalg.solve(np.eye(len(w))-w,u)
    return h,c@h

def gradients(w,c,u,y,method,steps=32,alpha_override=None):
    n=len(w);a=np.eye(n)-w;h,p=forward(w,c,u);batch=y.shape[1]
    if method in ['implicit','readout']:
        lam=-np.linalg.solve(a.T,c.T@(p-y))
        return -lam@h.T/batch,(p-y)@h.T/batch,0.
    f=np.eye(n)-w.T*(w!=0) if method=='masked' else a.T
    # Actual measured mask must be supplied for zero-valued projected weights.
    # Global MASK is immutable graph support, set before each fit/check.
    if method=='masked':f=np.eye(n)-w.T*MASK
    lam=np.zeros_like(h)
    alpha=(0. if method=='pc' else .2) if alpha_override is None else alpha_override
    for k in range(steps):
        r=a@h-u
        h=h-.1*(c.T@(c@h-y)+f@(lam+r))
        r=a@h-u
        if k<steps-1:lam=lam+alpha*r
    q=lam+r
    return -q@h.T/batch,(c@h-y)@h.T/batch,float(np.sqrt(np.mean(r*r)))

def project(w,c,mask,signs,out):
    w=np.where(signs[None,:]!=0,signs[None,:]*np.maximum(w*signs[None,:],0),w)*mask
    w*=min(1.,.8/max(np.linalg.norm(w,2),1e-15))
    c=c*out
    c*=min(1.,2./max(np.linalg.norm(c,2),1e-15))
    return w,c

def metrics(w,c,u,y):
    _,p=forward(w,c,u)
    return float(.5*np.mean(np.sum((p-y)**2,axis=0))),float(np.mean(p.argmax(0)==y.argmax(0)))

def verify():
    global MASK
    w,c,b,MASK,signs,out=initialize(99)
    rng=np.random.default_rng(44);u=rng.normal(size=(len(w),3));y=rng.normal(size=(10,3))
    gw,gc,_=gradients(w,c,u,y,'implicit')
    errs=[]
    for kind in ['w','c']:
        d=rng.normal(size=w.shape if kind=='w' else c.shape)*(MASK if kind=='w' else out)
        d/=np.linalg.norm(d);eps=1e-5
        plus=metrics(w+eps*d,c,u,y)[0] if kind=='w' else metrics(w,c+eps*d,u,y)[0]
        minus=metrics(w-eps*d,c,u,y)[0] if kind=='w' else metrics(w,c-eps*d,u,y)[0]
        fd=(plus-minus)/(2*eps);an=np.sum((gw if kind=='w' else gc)*d)
        err=abs(fd-an)/max(abs(fd),abs(an),1e-10);assert err<1e-5
        errs.append(err)
    gp=gradients(w,c,u,y,'pc');ga=gradients(w,c,u,y,'alm',alpha_override=0.)
    assert all(np.allclose(a,b,atol=1e-12) for a,b in zip(gp,ga))
    ww,cc=project(w*10,c*10,MASK,signs,out)
    assert np.linalg.norm(ww,2)<=.800000001 and np.linalg.norm(cc,2)<=2.00000001
    assert np.all(ww[~MASK]==0) and np.all(cc[:,~out]==0)
    assert np.all(ww[:,signs!=0]*signs[signs!=0]>=0)
    return {'passed':True,'finite_difference_errors':errs,'alpha_zero_matches_pc':True,'projection_invariants':True}

def fit(seed,method,lr,split,x,y,phase,output):
    global MASK
    start=time.perf_counter();w,c,b,MASK,signs,out=initialize(seed);w0=w.copy()
    u=b@x;tr,va,te=split
    initial=metrics(w,c,u[:,va],y[:,va]);mom=[np.zeros_like(w),np.zeros_like(c)];var=[v.copy() for v in mom]
    step=0;curves=[]
    for epoch in range(20):
        order=np.random.default_rng(seed+1000+epoch).permutation(tr)
        for ix in np.array_split(order,6):
            gw,gc,res=gradients(w,c,u[:,ix],y[:,ix],method)
            gs=[gw*MASK,gc*out]
            if method=='readout':gs[0][:]=0
            step+=1
            for i,z in enumerate([w,c]):
                mom[i]=.9*mom[i]+.1*gs[i];var[i]=.999*var[i]+.001*gs[i]**2
                z-=lr*(mom[i]/(1-.9**step))/(np.sqrt(var[i]/(1-.999**step))+1e-8)
            w,c=project(w,c,MASK,signs,out)
        loss,acc=metrics(w,c,u[:,va],y[:,va]);assert np.isfinite(loss),'Nonfinite loss'
        curves.append(dict(phase=phase,seed=seed,method=method,lr=lr,epoch=epoch+1,validation_loss=loss,validation_accuracy=acc,residual=res))
    probe=tr[:32]
    gw,gc,res=gradients(w,c,u[:,probe],y[:,probe],method)
    ew,ec,_=gradients(w,c,u[:,probe],y[:,probe],'implicit')
    g=gw*MASK;ref=ew*MASK
    row=dict(phase=phase,seed=seed,method=method,lr=lr,initial_validation_loss=initial[0],
       validation_loss=loss,validation_accuracy=acc,w_change=float(np.linalg.norm(w-w0)),
       gradient_error=float(np.linalg.norm(g-ref)/max(np.linalg.norm(ref),1e-15)),
       gradient_cosine=float(np.sum(g*ref)/max(np.linalg.norm(g)*np.linalg.norm(ref),1e-15)),
       seconds=time.perf_counter()-start,status='ok')
    if phase=='evaluation':row['test_loss'],row['test_accuracy']=metrics(w,c,u[:,te],y[:,te])
    stem=f'{phase}_{seed}_{method}_{lr}'
    dump(output/'runs'/(stem+'.json'),row)
    pd.DataFrame(curves).to_csv(output/'runs'/(stem+'.csv'),index=False)
    print(json.dumps(row),flush=True)
    return row

def main():
    out=HERE/'results';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    checks=verify()
    data=load_digits();x=(data.data/16).T;y=np.eye(10)[data.target].T
    ids=np.arange(len(data.target))
    tr,other=train_test_split(ids,train_size=900,random_state=45100,stratify=data.target)
    va,te=train_test_split(other,train_size=300,random_state=45100,stratify=data.target[other])
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    checks['disjoint_splits']=True
    dump(out/'verification.json',checks)
    dump(out/'split.json',{'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()})
    audit=json.loads((ROOT/'experiments/feedback_audit/results/manifest.json').read_text())
    raw=ROOT/'data/raw/malecns-v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    assert digest(raw)==audit['raw_sha256']
    edges=ROOT/'experiments/credit_rebuild/all_selected_edges.csv'
    assert digest(edges)==audit['hashes'][str(edges.relative_to(ROOT))]
    manifest={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
      'raw_sha256':audit['raw_sha256'],'dataset_sha256':hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest(),
      'hashes':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),HERE/'PROTOCOL.md',edges,out/'split.json']}}
    dump(out/'manifest_before_training.json',manifest)
    rows=[]
    for seed in range(46000,46003):
        for method in METHODS:
            for lr in [.001,.003]:rows.append(fit(seed,method,lr,(tr,va,te),x,y,'development',out))
    dev=pd.DataFrame(rows);dev.to_csv(out/'development.csv',index=False)
    selected={}
    for method in METHODS:
        selected[method]=float(dev[dev.method==method].groupby('lr').validation_loss.mean().idxmin())
    dump(out/'selected_before_evaluation.json',selected)
    evaluation=[]
    for seed in range(47000,47005):
        for method in METHODS:evaluation.append(fit(seed,method,selected[method],(tr,va,te),x,y,'evaluation',out))
    df=pd.DataFrame(evaluation);df.to_csv(out/'evaluation.csv',index=False)
    df.groupby('method')[['test_loss','test_accuracy','validation_loss','validation_accuracy','gradient_error','gradient_cosine','w_change','seconds']].agg(['mean','std']).to_csv(out/'summary.csv')
    pairs=[]
    for other in ['pc','implicit','masked','readout']:
        aa=df[df.method=='alm'].set_index('seed').test_loss
        bb=df[df.method==other].set_index('seed').test_loss
        delta=aa-bb;se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        pairs.append({'comparison':'alm-minus-'+other,'mean_test_loss_difference':float(delta.mean()),'paired_differences':delta.to_dict(),
          'exploratory_95pct_t_interval':[float(delta.mean()-student_t.ppf(.975,4)*se),float(delta.mean()+student_t.ppf(.975,4)*se)]})
    dump(out/'paired_comparisons.json',pairs)
    gates={}
    for method in ['implicit','alm']:
        d=df[df.method==method]
        gates[method+'_learnability']=bool(d.validation_accuracy.mean()>=.70 and (1-d.validation_loss/d.initial_validation_loss).mean()>=.2)
    gates['alm_over_readout']=bool(df[df.method=='alm'].validation_loss.mean()<=.99*df[df.method=='readout'].validation_loss.mean())
    gates['novel_method_demonstrated']=False;gates['biological_task_validated']=False
    dump(out/'gates.json',gates)
    print('GATES',gates,flush=True)

if __name__=='__main__':main()
