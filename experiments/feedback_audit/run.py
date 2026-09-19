"""Frozen-weight linear credit diagnostic; no model training."""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    out=HERE/'results'
    out.mkdir(exist_ok=False)
    nodes=pd.read_csv(ROOT/'experiments/malecns_v1_real/assignments.csv').sort_values('body_id')
    edges=pd.read_csv(ROOT/'experiments/credit_rebuild/all_selected_edges.csv')
    n=len(nodes); ids=nodes.body_id.to_numpy()
    raw=ROOT/'data/raw/malecns-v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    expected=json.loads((ROOT/'experiments/credit_rebuild/manifest.json').read_text())['raw_sha256'][raw.name]
    observed=sha(raw)
    assert observed==expected, 'Raw connectivity checksum mismatch'
    chunks=[]
    with pa.memory_map(str(raw),'r') as source:
        reader=pa.ipc.open_file(source)
        for idx in range(reader.num_record_batches):
            batch=reader.get_batch(idx)
            pre,post,weight=[batch.column(i).to_numpy() for i in range(3)]
            keep=np.isin(pre,ids)&np.isin(post,ids)&(weight>=5)
            if keep.any(): chunks.append(np.column_stack([pre[keep],post[keep],weight[keep]]))
    actual=pd.DataFrame(np.concatenate(chunks),columns=['pre','post','synapses']).astype('int64')
    expected_edges=edges[['pre','post','synapses']].sort_values(['pre','post']).to_numpy()
    assert np.array_equal(actual.sort_values(['pre','post']).to_numpy(),expected_edges), 'Selected-edge mismatch'
    assert n==132 and len(edges)==2663
    syn=np.zeros((n,n))
    syn[np.searchsorted(ids,edges.post),np.searchsorted(ids,edges.pre)]=edges.synapses
    s=syn>0; eye=np.eye(n)
    c=eye[nodes.layer.to_numpy()==3]
    b=c.T@c
    manifest={'python':platform.python_version(),'numpy':np.__version__,'neurons':n,'edges':int(s.sum()),
              'missing_reverse_edges':int((s.T & ~s).sum()),'raw_sha256':observed,'raw_selected_edges_verified':True,'hashes':{}}
    for p in [HERE/'PROTOCOL.md',Path(__file__),ROOT/'experiments/credit_rebuild/all_selected_edges.csv',ROOT/'experiments/malecns_v1_real/assignments.csv']:
        manifest['hashes'][str(p.relative_to(ROOT))]=sha(p)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cases=[]; predictions=[]; checks=[]
    for seed in range(42000,42005):
        rng=np.random.default_rng(seed)
        signs=nodes.sign.to_numpy().copy()
        signs[signs==0]=rng.choice([-1,1],size=(signs==0).sum())
        base=np.log1p(syn)*rng.uniform(.5,1.5,(n,n))*signs[None,:]
        u=rng.normal(size=n); y=rng.normal(size=len(c))
        for gain in [.4,.8]:
            w=base*gain/np.linalg.norm(base,2); a=eye-w
            hstar=np.linalg.solve(a,u)
            lamstar=-np.linalg.solve(a.T,c.T@(c@hstar-y))
            truth=-np.outer(lamstar,hstar)*s
            direction=rng.normal(size=(n,n))*s; direction/=np.linalg.norm(direction)
            def loss(v):
                hh=np.linalg.solve(eye-v,u)
                return .5*np.sum((c@hh-y)**2)
            eps=1e-5
            fd=(loss(w+eps*direction)-loss(w-eps*direction))/(2*eps)
            analytic=np.sum(truth*direction)
            fd_err=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-10)
            assert fd_err<1e-5,('finite_difference',seed,gain,fd_err)
            checks.append({'seed':seed,'gain':gain,'finite_difference_error':fd_err})
            for method in ['pc','alm_exact','alm_masked']:
                f=eye-w.T*s if method=='alm_masked' else a.T
                eta=.2; alpha=0. if method=='pc' else .2
                k=eye-eta*(b+f@a)
                m=np.block([[k,-eta*f],[alpha*a@k,eye-alpha*eta*a@f]])
                if method=='pc':
                    hs=np.linalg.solve(b+f@a,c.T@y+f@u); ls=np.zeros(n)
                    spectral=float(max(abs(np.linalg.eigvals(k))))
                else:
                    hs=hstar; ls=-np.linalg.solve(f,c.T@(c@hs-y))
                    spectral=float(max(abs(np.linalg.eigvals(m))))
                target=np.r_[hs,ls]
                ginf=-np.outer(ls+a@hs-u,hs)*s
                bias=float(np.linalg.norm(ginf-truth)/np.linalg.norm(truth))
                predictions.append(dict(seed=seed,gain=gain,method=method,spectral_radius=spectral,
                    predicted_stable=spectral<1,predicted_gradient_bias=bias))
                cases.append((seed,gain,method,a,f,hstar,target,m,truth,u,y,alpha,spectral))
    # Predictions and protocol are frozen before executing any inference trajectory.
    pd.DataFrame(predictions).to_csv(out/'predictions_before_inference.csv',index=False)
    manifest['predictions_sha256']=sha(out/'predictions_before_inference.csv')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for seed,gain,method,a,f,hstar,target,m,truth,u,y,alpha,spectral in cases:
        h=hstar.copy(); lam=np.zeros(n); predicted=np.r_[h,lam]-target
        max_rec=0.
        for t in range(1,4097):
            r=a@h-u
            h=h-.2*(c.T@(c@h-y)+f@(lam+r))
            # Endpoint gradient uses pre-dual multiplier, as in official default.
            credit=lam+a@h-u
            lam=lam+alpha*(a@h-u)
            predicted=m@predicted
            recurrence=np.linalg.norm(np.r_[h,lam]-target-predicted)/max(1.,np.linalg.norm(target))
            max_rec=max(max_rec,float(recurrence))
            assert recurrence<1e-9,('recurrence',seed,gain,method,t,recurrence)
            if t in [8,32,128,512,4096]:
                grad=-np.outer(credit,h)*s
                error=float(np.linalg.norm(grad-truth)/np.linalg.norm(truth))
                cosine=float(np.sum(grad*truth)/max(np.linalg.norm(grad)*np.linalg.norm(truth),1e-30))
                rows.append(dict(seed=seed,gain=gain,method=method,steps=t,gradient_error=error,
                    gradient_cosine=cosine,residual=float(np.linalg.norm(a@h-u)),
                    fixed_point_distance=float(np.linalg.norm(np.r_[h,lam]-target)/max(1.,np.linalg.norm(target))),
                    recurrence_error=float(recurrence)))
                if t==4096 and method=='alm_exact' and spectral<1:
                    assert error<1e-5,('exact_gradient_gate',seed,gain,error)
        checks.append(dict(seed=seed,gain=gain,method=method,max_recurrence_error=max_rec))
    pd.DataFrame(rows).to_csv(out/'trajectories.csv',index=False)
    pd.DataFrame(rows).groupby(['gain','method','steps'])[['gradient_error','gradient_cosine','residual']].mean().to_csv(out/'summary.csv')
    (out/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
    print(pd.read_csv(out/'summary.csv').query('steps == 512 or steps == 4096').to_string(index=False))
    print(json.dumps({k:v for k,v in manifest.items() if k!='hashes'},indent=2))

if __name__=='__main__': main()
