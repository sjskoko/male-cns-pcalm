"""Render an explanatory video from measured demo curves (not a screen recording)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/launch'
OUT.mkdir(parents=True, exist_ok=True)
results = json.loads((ROOT / 'examples/demo_output/demo_results.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#e5edf5',
                     'axes.labelcolor': '#e5edf5', 'xtick.color': '#9bacc1',
                     'ytick.color': '#9bacc1', 'axes.edgecolor': '#536477'})
fig = plt.figure(figsize=(12.8, 7.2), facecolor='#0b1220')
fig.text(.07, .9, 'Can a network wired from a fruit fly learn?', fontsize=25, weight='bold')
fig.text(.07, .84, 'MaleCNS × PC-ALM  |  A reproducible local-learning sandbox', fontsize=15)
fig.text(.07, .75, '132 neurons   /   1,001 retained connections', fontsize=20, color='#5eead4')
ax = fig.add_axes((.1, .24, .8, .43), facecolor='#111c2d')
lines = {}
for method, color in zip(results, ['#b5c1d1', '#5eead4', '#fbbe65']):
    values = [h['validation_mse'] for h in results[method]['history']]
    lines[method], = ax.plot(range(1,21), values, color=color, label=method.upper(), lw=2.8)
ax.set(xlabel='Epoch', ylabel='Validation MSE', xlim=(1,20))
ax.legend(facecolor='#111c2d', edgecolor='#536477', labelcolor='white')
fig.text(.07, .14, 'Measured demo curves: one seed, 20 epochs, synthetic targets. Not a brain simulation.', fontsize=12)
fig.text(.07, .09, 'Full study: PC-ALM learned, but did not outperform PC in the tested settings.', fontsize=12)
fig.text(.07, .04, 'github.com/sjskoko/male-cns-pcalm', fontsize=14, color='#5eead4')
fig.savefig(OUT / 'launch.png', dpi=100, facecolor=fig.get_facecolor())
def update(frame):
    n = min(20, 1 + int(frame / 10))
    for method, line in lines.items():
        values = [h['validation_mse'] for h in results[method]['history']]
        line.set_data(list(range(1,n+1)), values[:n])
    return list(lines.values())
animation = FuncAnimation(fig, update, frames=450, interval=100, blit=False)
animation.save(OUT / 'demo.mp4', writer=FFMpegWriter(fps=10, bitrate=650))
