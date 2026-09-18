"""Independent, resumable NumPy PC-ALM benchmark; run from this directory."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import time
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ['bp', 'pc', 'pcalm', 'tp', 'shuffle']
TASKS = ['nonlinear', 'linear']
HASHES = {
    'body-annotations-male-cns-v1.0-minconf-0.5.feather': '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2',
    'body-neurotransmitters-male-cns-v1.0.feather': '95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621',
    'connectome-weights-male-cns-v1.0-minconf-0.5.feather': 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1',
}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def prepare():
    raw = ROOT/'data/raw/malecns-v1.0'
    hashes = {f: digest(raw/f) for f in HASHES}
    assert hashes == HASHES, 'Raw data checksum mismatch'
    nodes = pd.read_csv(ROOT/'experiments/malecns_v1_real/assignments.csv')
    ids = np.sort(nodes.body_id.to_numpy())
    stage_map = dict(zip(nodes.body_id, nodes.layer))
    stages = np.array([stage_map[i] for i in ids])
    chunks = []
    with pa.memory_map(str(raw/'connectome-weights-male-cns-v1.0-minconf-0.5.feather'), 'r') as source:
        reader = pa.ipc.open_file(source)
        for k in range(reader.num_record_batches):
            b = reader.get_batch(k)
            pre, post, w = [b.column(i).to_numpy() for i in range(3)]
            a, c = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            a, c = np.minimum(a, len(ids)-1), np.minimum(c, len(ids)-1)
            keep = (ids[a]==pre)&(ids[c]==post)&(w>=5)
            if keep.any():
                chunks.append(np.column_stack([pre[keep],post[keep],w[keep],stages[a[keep]],stages[c[keep]]]))
    edges = np.concatenate(chunks).astype(np.int64)
    pd.DataFrame(edges, columns=['pre','post','synapses','pre_layer','post_layer']).to_csv(HERE/'all_selected_edges.csv',index=False)
    arrays = {}
    for layer in range(3):
        src = nodes[nodes.layer==layer].sort_values('body_id')
        dst = nodes[nodes.layer==layer+1].sort_values('body_id')
        syn = np.zeros((len(src),len(dst)))
        e = edges[(edges[:,3]==layer)&(edges[:,4]==layer+1)]
        syn[np.searchsorted(src.body_id,e[:,0]),np.searchsorted(dst.body_id,e[:,1])] = e[:,2]
        arrays[f'm{layer}'] = syn>0
        arrays[f's{layer}'] = np.broadcast_to(src.sign.to_numpy()[:,None], syn.shape)
        arrays[f'w{layer}'] = syn
    np.savez_compressed(HERE/'circuit.npz',**arrays)
    manifest = {'raw_sha256':hashes,'assignment_sha256':digest(ROOT/'experiments/malecns_v1_real/assignments.csv'),
                'neurons':len(nodes),'edges':sum(int(arrays[f'm{i}'].sum()) for i in range(3)),
                'all_selected_edges':len(edges),'discarded_nonadjacent_edges':int(np.sum(edges[:,4]-edges[:,3]!=1)),
                'layer_sizes':[int((nodes.layer==i).sum()) for i in range(4)],
                'unknown_sign_neurons':int((nodes.sign==0).sum())}
    dump(HERE/'manifest.json',manifest)
    print(manifest,flush=True)

def circuit():
    a=np.load(HERE/'circuit.npz')
    return [a[f'm{i}'] for i in range(3)], [a[f's{i}'] for i in range(3)]

def init(masks, signs, seed):
    rng=np.random.default_rng(seed)
    weights=[]
    for m,s in zip(masks,signs):
        w=rng.normal(size=m.shape)/np.sqrt(np.maximum(m.sum(0),1))[None,:]
        w=np.where(s!=0,np.abs(w)*s,w)*m
        weights.append(w)
    return weights

def forward(w,x):
    h=[x]
    for v in w[:-1]: h.append(np.tanh(h[-1]@v))
    return h, h[-1]@w[-1]

def bp(w,x,y):
    h,p=forward(w,x)
    delta=p-y
    g=[None]*len(w)
    g[-1]=h[-1].T@delta/len(x)
    for i in reversed(range(len(w)-1)):
        delta=(delta@w[i+1].T)*(1-h[i+1]**2)
        g[i]=h[i].T@delta/len(x)
    return g

def scales(masks, method, seed):
    s=[np.ones(m.shape[1]) for m in masks[:-1]]
    if method in ('tp','shuffle'):
        s=[np.clip((np.maximum(m.sum(0),1)/np.median(m.sum(0)[m.sum(0)>0]))**.25,.75,1.5) for m in masks[:-1]]
    if method=='shuffle':
        rng=np.random.default_rng(seed+80000)
        s=[rng.permutation(v) for v in s]
    return s

def signals(w,h,dual,s):
    predictions=[np.tanh(h[i]@w[i]) for i in range(len(w)-1)]
    residual=[h[i+1]-predictions[i] for i in range(len(predictions))]
    q=[dual[i]/s[i]+residual[i]/s[i]**2 for i in range(len(residual))]
    return predictions,residual,q

def local(w,x,y,s,eta=.1,alpha=.2,budget=8):
    h,_=forward(w,x)
    dual=[np.zeros_like(v) for v in h[1:]]
    for t in range(budget):
        p,r,q=signals(w,h,dual,s)
        dh=[]
        for i in range(len(q)):
            if i==len(q)-1: grad=q[i]+(h[-1]@w[-1]-y)@w[-1].T
            else: grad=q[i]-(q[i+1]*(1-p[i+1]**2))@w[i+1].T
            dh.append(grad)
        h=[x]+[v-eta*g for v,g in zip(h[1:],dh)]
        if t<budget-1:
            _,r,_=signals(w,h,dual,s)
            dual=[v+alpha*e/scale for v,e,scale in zip(dual,r,s)]
    p,r,q=signals(w,h,dual,s)
    g=[-h[i].T@(q[i]*(1-p[i]**2))/len(x) for i in range(len(q))]
    g.append(h[-1].T@(h[-1]@w[-1]-y)/len(x))
    return g,float(np.sqrt(np.mean(np.concatenate(r,axis=1)**2)))

def data(masks,signs,seed,task):
    rng=np.random.default_rng(seed+10000)
    x=rng.normal(size=(1024,masks[0].shape[0]))
    if task=='nonlinear': y=forward(init(masks,signs,seed+20000),x)[1]
    else:
        teacher=np.random.default_rng(seed+20000).normal(size=(x.shape[1],masks[-1].shape[1]))/np.sqrt(x.shape[1])
        y=x@teacher
    noisy=y[:512]+rng.normal(size=y[:512].shape)*(.01*y[:512].std())
    return x[:512],noisy,x[512:768],y[512:768],x[768:],y[768:]

def cosine(a,b,masks):
    a=np.concatenate([v[m] for v,m in zip(a,masks)])
    b=np.concatenate([v[m] for v,m in zip(b,masks)])
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))

def fit(masks,signs,task,seed,method,params,epochs=60,final=False):
    x,y,xv,yv,xt,yt=data(masks,signs,seed,task)
    w=init(masks,signs,seed+30000)
    s=scales(masks,method,seed)
    lr,eta,alpha=params
    m=[np.zeros_like(v) for v in w]; v=[a.copy() for a in m]
    rng=np.random.default_rng(seed+40000)
    history=[]; alignment=[]; residual=0.; step=0
    if final and method!='bp':
        ref=bp(w,x[:64],y[:64])
        for budget in [1,2,4,8,16,32]:
            g,r=local(w,x[:64],y[:64],s,eta,alpha,budget)
            alignment.append({'task':task,'seed':seed,'method':method,'budget':budget,'hidden_cosine':cosine(g[:-1],ref[:-1],masks[:-1]),'all_cosine':cosine(g,ref,masks),'residual':r})
    started=time.perf_counter()
    for epoch in range(epochs):
        idx=rng.permutation(len(x))
        for j in range(0,len(x),64):
            b=idx[j:j+64]; step+=1
            if method=='bp': g=bp(w,x[b],y[b])
            else: g,residual=local(w,x[b],y[b],s,eta,alpha)
            for i in range(len(w)):
                g[i]=g[i]*masks[i]
                m[i]=.9*m[i]+.1*g[i]; v[i]=.999*v[i]+.001*g[i]**2
                w[i]-=lr*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
                w[i]=np.where(signs[i]!=0,signs[i]*np.maximum(w[i]*signs[i],0),w[i])*masks[i]
        loss=float(np.mean((forward(w,xv)[1]-yv)**2))
        if not np.isfinite(loss): raise FloatingPointError((task,seed,method,params,epoch))
        if final: history.append({'task':task,'seed':seed,'method':method,'epoch':epoch+1,'validation_mse':loss})
    row={'task':task,'seed':seed,'method':method,'lr':lr,'eta':eta,'alpha':alpha,'validation_mse':loss,'runtime_seconds':time.perf_counter()-started,'residual':residual}
    if final:
        pred=forward(w,xt)[1]
        row['test_mse']=float(np.mean((pred-yt)**2))
        row['r2']=float(1-np.sum((pred-yt)**2)/np.sum((yt-yt.mean(0))**2))
        perturb=np.random.default_rng(seed+50000).normal(size=xt.shape)*.2
        row['noisy_input_mse']=float(np.mean((forward(w,xt+perturb)[1]-yt)**2))
        row['initial_test_mse']=float(np.mean((forward(init(masks,signs,seed+30000),xt)[1]-yt)**2))
    return row,history,alignment

def candidates(method):
    if method=='bp': return [(lr,.1,0.) for lr in [.0001,.0003,.0005,.001,.002,.003,.006,.01]]
    if method=='pc': return [(lr,eta,0.) for lr in [.0005,.001,.003,.006] for eta in [.05,.1]]
    return list(itertools.product([.001,.003],[.05,.1],[.2,.5]))

def tune():
    masks,signs=circuit(); rows=[]; selected={}
    for task in TASKS:
        selected[task]={}
        for method in METHODS:
            means=[]
            for ci,params in enumerate(candidates(method)):
                values=[]
                for seed in range(1100,1105):
                    path=HERE/f'checkpoints/dev_{task}_{method}_{ci}_{seed}.json'
                    if path.exists(): row=json.loads(path.read_text())
                    else:
                        row=fit(masks,signs,task,seed,method,params)[0]; dump(path,row)
                    rows.append(row); values.append(row['validation_mse'])
                means.append(np.mean(values))
            chosen=candidates(method)[int(np.argmin(means))]
            selected[task][method]=chosen
            print('selected',task,method,chosen,min(means),flush=True)
            pd.DataFrame(rows).to_csv(HERE/'development.csv',index=False)
    dump(HERE/'frozen.json',{'selected':selected,'protocol_sha256':digest(HERE/'PROTOCOL.md'),'code_sha256':digest(Path(__file__)),'circuit_sha256':digest(HERE/'circuit.npz'),'development_seeds':list(range(1100,1105)),'final_seeds':list(range(1200,1220))})

def run():
    frozen=json.loads((HERE/'frozen.json').read_text())
    assert frozen['code_sha256']==digest(Path(__file__))
    masks,signs=circuit()
    for task in TASKS:
        for seed in range(1200,1220):
            for method in METHODS:
                path=HERE/f'checkpoints/final_{task}_{method}_{seed}.json'
                if path.exists(): continue
                row,h,a=fit(masks,signs,task,seed,method,frozen['selected'][task][method],final=True)
                dump(path,{'row':row,'history':h,'alignment':a})
            print('completed',task,seed,flush=True)
    report()

def report():
    rec=[json.loads(p.read_text()) for p in sorted((HERE/'checkpoints').glob('final_*.json'))]
    rows=pd.DataFrame([r['row'] for r in rec]); assert len(rows)==200
    hist=pd.DataFrame([h for r in rec for h in r['history']])
    align=pd.DataFrame([a for r in rec for a in r['alignment']])
    rows.to_csv(HERE/'trials.csv',index=False); hist.to_csv(HERE/'curves.csv',index=False); align.to_csv(HERE/'alignment.csv',index=False)
    summary=rows.groupby(['task','method']).agg(mse=('test_mse','mean'),sd=('test_mse','std'),r2=('r2','mean'),initial_mse=('initial_test_mse','mean'),seconds=('runtime_seconds','mean'),noisy_mse=('noisy_input_mse','mean')).reset_index()
    summary.to_csv(HERE/'summary.csv',index=False)
    comparisons=[]; rng=np.random.default_rng(551)
    for task in TASKS:
        pivot=rows[rows.task==task].pivot(index='seed',columns='method',values='test_mse')
        for left,right in [('pc','pcalm'),('pcalm','tp'),('shuffle','tp')]:
            delta=(pivot[left]-pivot[right]).to_numpy()
            boot=rng.choice(delta,(20000,len(delta)),replace=True).mean(1)
            comparisons.append({'task':task,'left':left,'right':right,'delta_mse':float(delta.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'p':float(wilcoxon(delta,alternative='two-sided').pvalue) if np.any(delta) else 1.,'wins':int((delta>0).sum()),'reduction_percent':float(100*delta.mean()/pivot[left].mean())})
    order=np.argsort([c['p'] for c in comparisons]); last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,(len(order)-rank)*comparisons[i]['p']));comparisons[i]['holm_p']=last
    dump(HERE/'comparisons.json',comparisons)
    dump(HERE/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'code_sha256':digest(Path(__file__))})
    lines=['# Fresh experiment results','', '200 final fits; 2 tasks × 5 methods × 20 paired seeds. Development: 400 fits.','',summary.to_markdown(index=False),'','## Paired comparisons','',pd.DataFrame(comparisons).to_markdown(index=False),'','## Scope','', 'Real MaleCNS feedforward projection; synthetic targets. No whole-CNS, behavior, topology-efficiency or publication-readiness claim. Positive delta favors right. Two-sided Wilcoxon; Holm over six comparisons. CI are pointwise paired bootstrap intervals.','', 'Frozen settings: frozen.json. Full seed-level evidence: trials.csv; alignment.csv; curves.csv.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','tune','run','report']);args=parser.parse_args()
    globals()[args.action]()
