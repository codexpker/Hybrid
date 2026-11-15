"""Immutable named views of already archived G36 score fields, without forward.

For a same-weight scoring ablation, preserve the original bank and bind each
derived field to its source hash. This is not fitting or choosing a readout.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(source, output, field):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(source.glob('*__*.npz')):
        original_sha = sha(path)
        with np.load(path, allow_pickle=False) as z:
            index = z['index'].copy()
            score = z[field].copy()
            assert score.shape == index.shape and np.isfinite(score).all()
            metadata = {k: z[k].copy() for k in
                ['available_index', 'min_input_index', 'max_input_index', 'max_condition_index']}
        target = output/path.name
        np.savez_compressed(target, index=index, score=score, **metadata)
        assert sha(path) == original_sha
        rows.append(dict(source=str(path), source_sha256=original_sha,
                         field=field, output=str(target), output_sha256=sha(target), n=len(index)))
    assert len(rows) == 6, 'One SELECT, one CAL and all four original QUERY records required'
    with (output/'VIEW_RECEIPT.json').open('x', encoding='utf-8') as f:
        json.dump(dict(status='MATERIALIZED_FIXED_FIELD', score_field=field, banks=rows,
            model_forward=0, fitting=0, source_module_sha256=sha(__file__),
            purpose='Predefined same-weight score view; no readout selection'), f, indent=2)
    return rows
