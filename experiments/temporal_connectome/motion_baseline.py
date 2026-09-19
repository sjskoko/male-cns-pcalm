"""Post-primary sanity control; fixed image matching, no tuning or neural learning."""
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import dataset


def predict(x):
    frames = x.reshape(len(x), x.shape[1], 4, 8)
    previous, following = frames[:-1], frames[1:]
    templates = [np.roll(previous, 1, axis=3), np.roll(previous, -1, axis=3),
                 .5*(previous+np.roll(previous, 1, axis=2)),
                 .5*(previous+np.roll(previous, -1, axis=2))]
    errors = np.stack([((v-following)**2).mean(axis=(0, 2, 3)) for v in templates], axis=1)
    return np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])[errors.argmin(1)]


if __name__ == '__main__':
    rows = []
    for seed in range(3100, 3110):
        xt, yt, _ = dataset(seed)[2]
        pred = predict(xt)
        rows.append(dict(seed=seed, accuracy=float(np.mean(np.all(pred == yt, axis=1))),
                         mse=float(np.mean((pred-yt)**2))))
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(__file__).parent/'results/motion_baseline.csv', index=False)
    print(frame[['accuracy', 'mse']].mean().to_dict())
