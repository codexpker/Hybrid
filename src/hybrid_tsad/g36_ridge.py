"""Fixed linear historical-response controls; no anomaly labels are read."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .g36_data import HISTORY, BLOCK


def windows(x):
    ends = np.arange(HISTORY + BLOCK - 1, len(x))
    history = np.stack([x[t-BLOCK-HISTORY+1:t-BLOCK+1] for t in ends])
    target = np.stack([x[t-BLOCK+1:t+1, :4] for t in ends])
    return ends, history, target


def run(data, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    with np.load(data / 'BaselineSystem__FIT.npz') as z:
        _, history, target = windows(z['x'])
    for name, channels in [('ridge_conditional', 9), ('ridge_history', 4)]:
        directory = output / name
        directory.mkdir()
        # One fixed regularization in FIT standardized units; no QUERY selection.
        model = Ridge(alpha=1.0, solver='cholesky')
        features = history[:, :, :channels].reshape(len(history), -1).astype(np.float64)
        model.fit(features, target.reshape(len(target), -1).astype(np.float64))
        np.savez(directory / 'model.npz', coef=model.coef_, intercept=model.intercept_, alpha=1.)
        for record in manifest['records']:
            if record['role'] not in ('SELECT', 'CAL', 'QUERY'):
                continue
            with np.load(data / record['path']) as z:
                ends, h, y = windows(z['x'])
                prediction = model.predict(h[:, :, :channels].reshape(len(h), -1)).reshape(y.shape)
                evidence = (prediction - y.astype(np.float64)) ** 2
                np.savez_compressed(directory / record['path'], index=z['index'][ends],
                                    score=evidence.mean((1, 2)), prediction=prediction,
                                    target=y, evidence=evidence)
        (directory / 'identity.json').write_text(json.dumps(dict(
            method=name, alpha=1., information='strict past only; current block absent',
            data_manifest_sha256=hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
            train_rows=len(features), history=HISTORY, block=BLOCK,
            coefficient_shape=list(model.coef_.shape)), indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.data, args.output)
