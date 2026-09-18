"""Scientific plots from saved CSV; no recomputation of experiment metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=pd.read_csv(root/'trials.csv')
align=pd.read_csv(root/'alignment.csv')
methods=['bp','pc','pcalm','tp','shuffle']
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,task in zip(axes,['nonlinear','linear']):
    pivot=rows[rows.task==task].pivot(index='seed',columns='method',values='test_mse')[methods]
    for _,r in pivot.iterrows(): ax.plot(range(5),r.values,color='gray',alpha=.25,lw=.8)
    for i,m in enumerate(methods): ax.scatter(np.full(len(pivot),i),pivot[m],s=12,label=m)
    ax.set_xticks(range(5),methods);ax.set_yscale('log');ax.set_ylabel('Test MSE');ax.set_title(task+' teacher (paired seeds)')
fig.tight_layout();fig.savefig(root/'paired_mse.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,task in zip(axes,['nonlinear','linear']):
    for m in methods[1:]:
        g=align[(align.task==task)&(align.method==m)].groupby('budget').hidden_cosine
        ax.errorbar(g.mean().index,g.mean(),yerr=1.96*g.sem(),marker='o',label=m,capsize=2)
    ax.set_xscale('log',base=2);ax.set_xlabel('Inference steps');ax.set_ylabel('Hidden-weight BP gradient cosine');ax.set_title(task+' teacher');ax.legend()
fig.tight_layout();fig.savefig(root/'gradient_alignment.png',dpi=180);plt.close(fig)
