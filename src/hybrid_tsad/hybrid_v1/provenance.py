"""Import actual SHA-pinned external classes without their heavyweight wrappers."""
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

TIMESFM_COMMIT = "e51928e27119cb17bebc005be2696b75e0a9e688"
LEFT_COMMIT = "3fadb4811797075e233076b9faceb0d8ec59b697"
TIMERADAR_COMMIT = "4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2"
TF_SHA = "881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317"
TIMESFM_HASHES = {
    "configs.py": "911e29ab7aa54b34dfa3ca5fe89b2564055b20da87ad490d2c63d64fa4fd3739",
    "dense.py": "74d3229d7128f9db3f4c653639d79ac4202dcbc531424394d334d604574ed6bd",
    "normalization.py": "0b0ae0e34810f9b9d5e2d1bea2dc2a9b12ec908cf61b53c5055c595b9f4bc013",
    "transformer.py": "fc23e20a6cb65c41080ed7a2afc5442f89760ce6756bcf10d186bd8b533382a6",
    "util.py": "a1f4837958e33211a623166a1d6e1902141fe89059203f8621742f70fe9a550f"}
LEFT_HASHES = {
    "Tools.py": "0ad478d91e227eda0b3d75708f137673ab489240d220b2c68ad4dd422344ad7c",
    "Utils.py": "262290e1690d0dfc29fe6cf5cbbc0b1a7fcd2c4663f87eb6dfa7cb54d0a14a4a"}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package(root, prefix, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        if not expected or file_sha(root / name) != expected:
            raise ValueError("External official identity mismatch: " + str(root / name))
    namespace = prefix + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(root)]
        package.__package__ = namespace
        sys.modules[namespace] = package
    return namespace


@dataclass(frozen=True)
class OfficialSources:
    timeradar_file: str
    left_root: str
    timesfm_root: str

    @classmethod
    def from_workspace(cls, root):
        root = Path(root)
        return cls(str(root / "docs/research_review_2026-10-27/source_snapshots/TimeRadar/models/TimeRadar.py"),
                   str(root / "literature/novelty_review/left_source_20261031/models/Left"),
                   str(root / "literature/reference_source_20261031/timesfm3/src/timesfm3/torch"))

    def timeradar(self):
        if file_sha(self.timeradar_file) != TF_SHA:
            raise ValueError("External TimeRadar identity mismatch")
        name = "_hybrid_v1_timeradar_" + TF_SHA[:12]
        spec = importlib.util.spec_from_file_location(name, self.timeradar_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def left(self):
        package = _package(self.left_root, "_hybrid_v1_left_", LEFT_HASHES)
        return importlib.import_module(package + ".Tools"), importlib.import_module(package + ".Utils")

    def timesfm(self):
        package = _package(self.timesfm_root, "_hybrid_v1_timesfm_", TIMESFM_HASHES)
        return tuple(importlib.import_module(package + "." + n) for n in ("configs", "dense", "transformer"))

    def manifest(self):
        paths = [(Path(self.timeradar_file), TF_SHA, TIMERADAR_COMMIT)]
        paths += [(Path(self.left_root) / n, h, LEFT_COMMIT) for n, h in LEFT_HASHES.items()]
        paths += [(Path(self.timesfm_root) / n, h, TIMESFM_COMMIT) for n, h in TIMESFM_HASHES.items()]
        result = []
        for path, expected, commit in paths:
            actual = file_sha(path)
            if actual != expected:
                raise ValueError("External source changed")
            result.append(dict(path=str(path.resolve()), sha256=actual, commit=commit))
        return result
