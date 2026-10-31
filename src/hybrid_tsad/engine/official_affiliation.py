"""Read-only adapter to the already audited external six-file implementation."""
import hashlib
import importlib.util
from pathlib import Path
import sys

from ..config import read_json, resolve_path, sha256

OFFICIAL_FILES={
    "_integral_interval.py":"8177e69eccb0ae4ea76b2f54f6799ae61758b5a103603fae3d1a3a8d331e7cda",
    "_single_ground_truth_event.py":"a04b5daa0fb0ef68893807c7ab5d4353b43f0d5231ae06ef0006b2e6cf939165",
    "__init__.py":"15c801cfb3a60060383c096e5afd2b01a57c7d65bb2a0a6fa012dddd417c01d7",
    "generics.py":"16743b204d24feb8ba9deaabceb7b0cf900520136e8a3b77e499448fcaafbfc9",
    "metrics.py":"1442b1dc6801e3c12db39c9b04a54e56ec29a5de2755eb2f5821f1183aa813e0",
    "_affiliation_zone.py":"3b289bf11cebe614afabe353c873de90ba9debf8831aef8f73532870dce01ba2"}


class OfficialAffiliationBlocked(ValueError):
    pass


def load_official_affiliation(identity_path):
    identity=read_json(identity_path)
    if identity.get("schema_version")!=1 or identity.get("files")!=OFFICIAL_FILES:
        raise OfficialAffiliationBlocked("BLOCKED: external official Affiliation six-file identity is not the audited version")
    license_info=identity.get("license",{})
    if (not license_info.get("status") or not license_info.get("source") or license_info.get("redistribute") is not False):
        raise OfficialAffiliationBlocked("BLOCKED: record existing license provenance/status; upstream redistribution is forbidden")
    directory=resolve_path(identity["directory"],identity_path)
    if not directory.is_dir() or {p.name for p in directory.glob("*.py")}!=set(OFFICIAL_FILES):
        raise OfficialAffiliationBlocked("BLOCKED: external official Affiliation files are missing or extra code is present")
    for name,expected in OFFICIAL_FILES.items():
        if sha256(directory/name)!=expected:
            raise OfficialAffiliationBlocked("BLOCKED: external official Affiliation SHA256 mismatch")
    namespace="_hybrid_official_aff_"+hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    for name in list(sys.modules):
        if name==namespace or name.startswith(namespace+"."):
            del sys.modules[name]
    dont_write=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec=importlib.util.spec_from_file_location(namespace,directory/"__init__.py",submodule_search_locations=[str(directory)])
        package=importlib.util.module_from_spec(spec);sys.modules[namespace]=package;spec.loader.exec_module(package)
        spec=importlib.util.spec_from_file_location(namespace+".metrics",directory/"metrics.py")
        module=importlib.util.module_from_spec(spec);sys.modules[namespace+".metrics"]=module;spec.loader.exec_module(module)
    except ImportError as exc:
        raise OfficialAffiliationBlocked(f"BLOCKED: official Affiliation dependency unavailable: {exc}") from exc
    finally:
        sys.dont_write_bytecode=dont_write
    if any(sha256(directory/name)!=expected for name,expected in OFFICIAL_FILES.items()):
        raise OfficialAffiliationBlocked("BLOCKED: external implementation changed during import")
    return module.pr_from_events,{"identity_sha256":sha256(identity_path),"files":OFFICIAL_FILES,
                                 "directory":str(directory),"license":license_info,"source_copied":False,"global_affiliation_defined":False}
