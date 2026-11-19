"""Optional external-source regression, not a bundled upstream redistribution.

Only annotations and the abstract base declaration are removed from the AST
to avoid importing unrelated optional torchtyping dependencies. The official
EnergyScore method bodies and original source-byte hash remain unchanged.
"""
import ast
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_tsad.conditional_response_energy import energy_score


OFFICIAL_SHA = "8d0e065143385c77e0e1f91970385bb80c72e9780643275e5385855e5e41f7bf"


def official_energy():
    default = Path(__file__).resolve().parents[2] / "literature/reference_source_20261119/jmlr_generative_energy/scoring_rules.official.py"
    path = Path(os.environ.get("HYBRID_JMLR_ENERGY_REFERENCE", default))
    if not path.exists():
        pytest.skip("External pinned JMLR source absent; set HYBRID_JMLR_ENERGY_REFERENCE")
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == OFFICIAL_SHA
    node = next(n for n in ast.parse(content).body if isinstance(n, ast.ClassDef) and n.name == "EnergyScore")
    node.bases = []
    for child in ast.walk(node):
        if isinstance(child, ast.arg):
            child.annotation = None
        elif isinstance(child, ast.FunctionDef):
            child.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"logging": logging, "torch": torch, "np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["EnergyScore"]


def test_per_sample_value_and_gradient_match_official_half_convention():
    reference = official_energy()
    stream = torch.Generator().manual_seed(509)
    samples = torch.randn(3, 4, 9, dtype=torch.float64, generator=stream, requires_grad=True)
    truth = torch.randn(3, 9, dtype=torch.float64, generator=stream)
    ours = energy_score(samples, truth)
    original = reference(beta=1, mean=True).estimate_score_batch(samples, truth) / 2
    torch.testing.assert_close(ours.mean(), original, rtol=1e-12, atol=1e-12)
    original_sum = reference(beta=1, mean=False).estimate_score_batch(samples, truth) / 2
    torch.testing.assert_close(ours.sum(), original_sum, rtol=1e-12, atol=1e-12)
    for b in range(len(samples)):
        point = reference().estimate_score_batch(samples[b:b+1], truth[b:b+1]) / 2
        torch.testing.assert_close(ours[b], point, rtol=1e-12, atol=1e-12)
    left = torch.autograd.grad(ours.mean(), samples, retain_graph=True)[0]
    right = torch.autograd.grad(original, samples)[0]
    torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
