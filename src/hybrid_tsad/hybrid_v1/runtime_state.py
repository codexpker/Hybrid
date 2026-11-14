"""State isolation for current validation calls, not a historical runtime patch.

Seeding torch.manual_seed also seeds every visible CUDA device.  A local fork
must instead seed only its saved generators.  No NumPy/Python RNG is consumed
or seeded by this helper; callers using those generators need their own guard.
"""
from contextlib import contextmanager


def cuda_devices(model):
    """Include buffers as well as parameters (including parameterless modules)."""
    return sorted({t.device.index for t in (*model.parameters(), *model.buffers())
                   if t.device.type == "cuda"})


@contextmanager
def isolated_rng(seed, devices=()):
    import torch
    devices = sorted(set(devices))
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.random.default_generator.manual_seed(int(seed))
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
        yield


@contextmanager
def evaluation_state(model, seed):
    """Restore heterogeneous module flags and used RNGs on all exit paths.

This guards mode/RNG, not arbitrary buffers mutated by a custom eval forward.
It intentionally does not replace the already-audited matched-U helper.
"""
    import torch
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with isolated_rng(seed, cuda_devices(model)), torch.no_grad():
            yield
    finally:
        for module, flag in flags:
            module.training = flag
