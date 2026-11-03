"""Label-free feature-major public-query conversion; no model or scaler imports.

The whole-file identity hash reads every byte, including embedded label bytes.
The CSV iterator consumes exactly header + feature_blocks * block_rows records,
and never requests the following label record. Only selected query values become
numbers. This is a processed benchmark axis, not native entity certification.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

OPERATION = "HYBRID_V1_PROCESSED_PUBLIC_QUERY_INPUT_DERIVATION"
RELEASE_STATUS = "PRIMARY_RELEASED_PUBLIC_QUERY_INPUT_DERIVATION"
PERMISSIONS = dict(wholefile_byte_hash=True, primary_query_numeric=True,
                   train_numeric_parse=False, nonprimary_numeric_parse=False,
                   label_numeric_parse=False, model_or_weights=False,
                   fit=False, query_forward=False, evaluation=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_config(config):
    """Metadata-only validation: never open the input, output or any label file."""
    if config.get("operation") != OPERATION:
        raise ValueError("Exact public-query derivation operation required")
    source, schema = config["input"], config["schema"]
    if (type(source["bytes"]) is not int or source["bytes"] <= 0
            or len(source["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["sha256"])):
        raise ValueError("Exact input bytes and SHA256 required")
    names, primary = schema["feature_names"], schema["primary_feature_indices"]
    if (schema["header"] != ["date", "data", "cols"]
            or not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n or n == "label" for n in names)
            or len(names) != len(set(names))
            or not isinstance(primary, list) or not primary
            or primary != sorted(set(primary))
            or any(type(i) is not int or not 0 <= i < len(names) for i in primary)
            or type(schema["block_rows"]) is not int
            or type(schema["split"]) is not int
            or not 0 < schema["split"] < schema["block_rows"]):
        raise ValueError("Explicit ordered feature blocks, selected indices and split required")
    if config.get("permissions") != PERMISSIONS:
        raise ValueError("Exact label-free derivation permissions required")
    if not config.get("output_dir") or not config.get("dataset"):
        raise ValueError("Dataset identity and fresh output path required")
    return schema


def _parse_features(stream, schema):
    """Never advance the CSV iterator past the last configured feature row."""
    rows = csv.reader(stream)
    try:
        header = next(rows)
    except StopIteration:
        raise ValueError("Missing CSV header") from None
    if header != schema["header"]:
        raise ValueError("CSV header mismatch")
    n, split = schema["block_rows"], schema["split"]
    selected = {i: j for j, i in enumerate(schema["primary_feature_indices"])}
    values = np.empty((n - split, len(selected)), dtype=np.float32)
    for feature, name in enumerate(schema["feature_names"]):
        for index in range(n):
            try:
                row = next(rows)
            except StopIteration:
                raise ValueError("Truncated configured feature blocks") from None
            if len(row) != 3 or row[0] != str(index + 1) or row[2] != name:
                raise ValueError("Feature block/date/column structure mismatch")
            if feature not in selected or index < split:
                continue
            try:
                number = float(row[1])
            except (ValueError, OverflowError):
                raise ValueError("Invalid selected query numeric token") from None
            if math.isinf(number) or (math.isfinite(number)
                                      and abs(number) > np.finfo(np.float32).max):
                raise ValueError("Inf or float32 overflow in selected query")
            values[index - split, selected[feature]] = number
    return values, np.isfinite(values)


def convert_query(config):
    """Scientific conversion API; production CLI additionally requires root release.

    Failure leaves the fresh output and a failure record for diagnosis. It never
    removes or overwrites an existing directory or retries the conversion.
    """
    schema = validate_config(config)
    output, source = Path(config["output_dir"]), Path(config["input"]["path"])
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    before = source.stat()
    if before.st_size != config["input"]["bytes"]:
        raise ValueError("Input byte size mismatch")
    started = time.monotonic()
    if sha256(source) != config["input"]["sha256"]:
        raise ValueError("Input SHA256 mismatch")
    for asset in config.get("metadata_assets", []):
        if sha256(asset["path"]) != asset["sha256"]:
            raise ValueError("Bound metadata SHA256 mismatch")
    output.mkdir(parents=True, exist_ok=False)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            values, observed = _parse_features(stream, schema)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Input changed during conversion")
        split, n = schema["split"], schema["block_rows"]
        arrays = dict(values=values, observed=observed,
                      processed_index=np.arange(split, n, dtype=np.int64),
                      test_index=np.arange(n - split, dtype=np.int64))
        npz = output / "query_inputs.npz"
        np.savez_compressed(npz, **arrays)
        array_identity = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                 sha256=hashlib.sha256(v.tobytes(order="C")).hexdigest())
                          for k, v in arrays.items()}
        columns = [schema["feature_names"][i] for i in schema["primary_feature_indices"]]
        _json(output / "INPUT_MANIFEST.json", dict(runs=[dict(
            domain=config["dataset"], entity="processed_anonymous_stream",
            run="processed_test_axis", role="QUERY", path="query_inputs.npz",
            format="npz", key="values", observed_key="observed",
            bounds=[0, n - split], expected_shape=list(values.shape),
            sha256=sha256(npz), original_start=0)]))
        _json(output / "COLUMN_SCHEMA.json", dict(
            feature_names=schema["feature_names"], primary_feature_indices=schema["primary_feature_indices"],
            primary_feature_names=columns, values_dtype="float32", observed_dtype="bool",
            numeric_conversion="float64 token parse then explicit float32; NaN retained",
            physical_sensor_roles_verified=False))
        _json(output / "AXIS_IDENTITY.json", dict(
            processed_bounds=[split, n], test_local_bounds=[0, n - split],
            native_entity_boundaries_certified=False,
            interpretation="anonymous specified processed benchmark row order; not physical time",
            candidate_boundaries_applied=False, readout_window_policy="separate future root contract"))
        report = dict(status="COMPLETED_PROCESSED_QUERY_INPUT_DERIVATION",
                      input=config["input"], schema=schema, arrays=array_identity,
                      wholefile_hash_includes_embedded_label_bytes=True,
                      label_records_requested_by_csv_iterator=0,
                      parsed_feature_records=n * len(schema["feature_names"]),
                      numeric_query_tokens=values.size, observed_values=int(observed.sum()),
                      missing_values=int((~observed).sum()), train_numeric_parse=False,
                      nonprimary_numeric_parse=False, label_numeric_parse=False,
                      scaler_fit=False, imputation=False, model_or_weights=False,
                      query_forward=False, evaluation=False, GPUh=0,
                      elapsed_seconds=time.monotonic() - started)
        _json(output / "CONVERSION_REPORT.json", report)
        _json(output / "ARTIFACT_MANIFEST.json", dict(files=[dict(
            path=p.name, bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(output.iterdir()) if p.is_file()]))
        return report
    except Exception as error:
        _json(output / "CONVERSION_FAILURE.json", dict(
            status="FAILED_NO_RETRY", error_type=type(error).__name__, error=str(error),
            label_numeric_parse=False, model_or_weights=False, GPUh=0))
        raise


def admit(config_path, release_path, trusted_sha256):
    """Verify exact named root authority before input stat/hash or conversion."""
    if not trusted_sha256 or sha256(release_path) != trusted_sha256:
        raise ValueError("Trusted root release SHA256 mismatch")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    release = json.loads(Path(release_path).read_text(encoding="utf-8"))
    validate_config(config)
    if (release.get("status") != RELEASE_STATUS or release.get("approved_by") != "root"
            or release.get("operation") != OPERATION
            or release.get("config_sha256") != sha256(config_path)
            or release.get("permissions") != PERMISSIONS
            or release.get("source_sha256") != config.get("source_sha256")):
        raise ValueError("Exact root operation/config/permissions/source binding required")
    package = Path(__file__).resolve().parents[1]
    closure = config.get("source_sha256", {})
    required = {"__init__.py", "hybrid_v1/__init__.py", "hybrid_v1/config.py",
                "hybrid_v1/provenance.py", "hybrid_v1/processed_query.py"}
    if not required.issubset(closure):
        raise ValueError("Complete package import source closure required")
    for name, expected in closure.items():
        path = package / name
        try:
            path.resolve().relative_to(package)
        except ValueError:
            raise ValueError("Source closure path escapes package")
        if sha256(path) != expected:
            raise ValueError("Source SHA256 mismatch")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--trusted-release-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
        print(json.dumps(dict(status="METADATA_PREVIEW_ONLY", input_opened=False, GPUh=0)))
        return
    if not args.release or not args.trusted_release_sha256:
        raise ValueError("Execute requires explicit release and externally trusted SHA256")
    result = convert_query(admit(args.config, args.release, args.trusted_release_sha256))
    print(json.dumps(dict(status=result["status"], numeric_query_tokens=result["numeric_query_tokens"], GPUh=0)))


if __name__ == "__main__":
    main()
