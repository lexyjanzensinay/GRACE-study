"""Executable temporal, group and provenance contracts for standalone GRACE.

Core validation requires only NumPy and pandas. Callable cost fingerprints use
cloudpickle lazily and fail closed if that optional serializer is unavailable.
Fingerprints use the explicit ``grace-typed-hash-v2`` representation; they are
not interchangeable with fingerprints produced by an older representation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

PRODUCT = "GRACE"
HASH_SCHEMA = "grace-typed-hash-v2"


def _hash_tree(value: Any, active: set[int]) -> Any:
    """Build a typed deterministic value tree; never replace objects with repr."""
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["pandas.NA"]
    if value is pd.NaT:
        return ["pandas.NaT"]
    if value is np.ma.masked:
        return ["numpy.masked"]
    if isinstance(value, np.generic):
        if value.dtype.names:
            return ["numpy-scalar", value.dtype.descr,
                    [[name, _hash_tree(value[name], active)] for name in value.dtype.names]]
        # Some extended floats contain ABI padding bytes that are not part of
        # the number and may vary between allocations. Canonical decimal output
        # retains the scalar's own precision without narrowing it to float64.
        if value.dtype.kind == "f" and value.dtype.itemsize > 8:
            return ["numpy-scalar", value.dtype.str,
                    np.format_float_scientific(value, unique=True)]
        if value.dtype.kind == "c" and value.dtype.itemsize > 16:
            return ["numpy-scalar", value.dtype.str,
                    _hash_tree(value.real, active), _hash_tree(value.imag, active)]
        if not value.dtype.hasobject:
            return ["numpy-scalar", value.dtype.str, value.tobytes().hex()]
        return ["numpy-scalar", value.dtype.str, _hash_tree(value.item(), active)]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        # Explicit tokens preserve missing/non-finite scientific array values
        # without emitting non-standard JSON or confusing them with strings.
        return ["float", value.hex() if np.isfinite(value) else str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Path):
        return ["path", str(value)]
    if isinstance(value, (datetime, date, time, pd.Timestamp)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, (timedelta, pd.Timedelta)):
        return [type(value).__name__, str(value)]
    ident = id(value)
    if ident in active:
        raise ValueError("Cyclic values cannot receive a stable provenance hash.")
    active.add(ident)
    try:
        if isinstance(value, np.ma.MaskedArray):
            # np.asarray discards a NumPy mask.  Both the observed support and
            # underlying raw values belong to an auditable data identity.
            return ["masked-array", _hash_tree(value.data, active),
                    _hash_tree(np.ma.getmaskarray(value), active),
                    _hash_tree(value.fill_value, active), bool(value.hardmask)]
        if isinstance(value, np.ndarray):
            dtype = value.dtype.descr if value.dtype.names else value.dtype.str
            if value.dtype.names:
                content = [[name, _hash_tree(value[name], active)] for name in value.dtype.names]
            elif ((value.dtype.kind == "f" and value.dtype.itemsize > 8)
                  or (value.dtype.kind == "c" and value.dtype.itemsize > 16)):
                content = [_hash_tree(item, active) for item in value.flat]
            elif value.dtype.hasobject:
                content = _hash_tree(value.tolist(), active)
            else:
                content = ["sha256-bytes", hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()]
            return ["ndarray", dtype, list(value.shape), content]
        if isinstance(value, Mapping):
            pairs = [[_hash_tree(k, active), _hash_tree(v, active)] for k, v in value.items()]
            pairs.sort(key=lambda item: json.dumps(item, separators=(",", ":"), ensure_ascii=False))
            return ["mapping", pairs]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [_hash_tree(v, active) for v in value]]
        if isinstance(value, (set, frozenset)):
            items = [_hash_tree(v, active) for v in value]
            items.sort(key=lambda item: json.dumps(item, separators=(",", ":"), ensure_ascii=False))
            return [type(value).__name__, items]
    finally:
        active.remove(ident)
    raise TypeError(f"Unsupported provenance value type: {type(value).__name__}; supply explicit values.")


def stable_hash(value: Any) -> str:
    """Hash a typed value tree, independent of mapping insertion order."""
    encoded = json.dumps([HASH_SCHEMA, _hash_tree(value, set())], separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_hash(value: Any) -> str:
    """Hash array shape, dtype and values, including typed object-array values."""
    return stable_hash(value if isinstance(value, np.ma.MaskedArray) else np.asarray(value))


def batch_hash(value: Any) -> str:
    """Include masks, timestamps and channel semantics when a masked batch is used."""
    if hasattr(value, "observation_mask"):
        return stable_hash({
            "values": array_hash(value.values), "observation_mask": array_hash(value.observation_mask),
            "timestamps": array_hash(value.timestamps),
            "channel_names": getattr(value, "channel_names", None),
            "original_ndim": getattr(value, "original_ndim", None),
        })
    return stable_hash([array_hash(row) for row in value])


def temporal_metadata(frame: pd.DataFrame) -> dict:
    """Extract prioritized temporal columns without case-insensitive ambiguity."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Temporal metadata must be a pandas DataFrame.")
    lower = {str(c).strip().lower(): c for c in frame.columns}
    if len(lower) != len(frame.columns):
        raise ValueError("Temporal metadata column names must be unique after normalization.")
    fields = {
        "episode_time": ("episode_time", "window_end", "timestamp", "date", "anchor"),
        "information_start": ("information_start", "window_start", "info_start"),
        "information_end": ("information_end", "future_date", "info_end"),
    }
    return {key: next((frame[lower[c]].to_numpy() for c in names if c in lower), None)
            for key, names in fields.items()}


def _validate_count(n: Any) -> int:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1:
        raise ValueError("Episode count must be a positive integer.")
    return int(n)


def utc_nanoseconds(values: Any, n: int, name: str) -> np.ndarray:
    """Return UTC int64 nanoseconds; naive datetimes mean UTC.

    Integer input is explicitly interpreted as Unix nanoseconds. Floating-point,
    boolean and sub-nanosecond timestamps are rejected rather than truncated.
    NumPy datetime unit multipliers are preserved before pandas conversion.
    Convert numeric timestamps expressed in other units before calling this API.
    """
    n = _validate_count(n)
    if values is None:
        raise ValueError(f"{name} must be one-dimensional and aligned with every episode.")
    try:
        vector = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional timestamp vector.") from exc
    if vector.ndim != 1 or len(vector) != n:
        raise ValueError(f"{name} must be one-dimensional and aligned with every episode.")
    items = []
    for value in values:
        if isinstance(value, (bool, np.bool_, float, np.floating)):
            raise ValueError(f"{name} may not contain boolean or floating-point timestamps.")
        if isinstance(value, np.datetime64):
            # Passing these scalars directly to pandas can lose dtype unit
            # multipliers (e.g. datetime64[2s]) or misread picoseconds as ns.
            # NumPy's own formatter retains the actual instant and precision.
            value = np.datetime_as_string(value, unit="auto")
        if isinstance(value, str):
            fractions = re.findall(r"[.,](\d{10,})", value)
            if any(any(digit != "0" for digit in fraction[9:]) for fraction in fractions):
                raise ValueError(f"{name} contains a timestamp finer than nanosecond precision.")
        items.append(value)
    try:
        parsed = pd.to_datetime(pd.Series(items, dtype=object), utc=True, errors="raise", format="mixed")
        if parsed.isna().any():
            raise ValueError("Missing timestamp")
        return parsed.dt.as_unit("ns").astype("int64").to_numpy(copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name} timestamps or values outside the nanosecond range.") from exc


def validate_temporal(n: int, episode_time: Any, information_start: Any, information_end: Any) -> dict:
    """Validate aligned information intervals and nondecreasing episode times."""
    result = {name: utc_nanoseconds(value, n, name) for name, value in [
        ("episode_time", episode_time), ("information_start", information_start),
        ("information_end", information_end),
    ]}
    if np.any(result["information_end"] < result["information_start"]):
        raise ValueError("information_end precedes information_start.")
    if np.any((result["episode_time"] < result["information_start"]) |
              (result["episode_time"] > result["information_end"])):
        raise ValueError("episode_time must lie within its information interval.")
    times = result["episode_time"]
    # Subtraction can overflow int64 for valid, widely separated timestamps.
    if np.any(times[1:] < times[:-1]):
        raise ValueError("Chronological episodes must be sorted by episode_time.")
    return result


def _categorical_vector(values: Any, name: str, *, integral: bool) -> np.ndarray:
    if isinstance(values, np.ma.MaskedArray):
        if values.dtype.names is not None:
            raise ValueError(f"{name} must contain scalar categorical identifiers, not structured records.")
        if np.ma.is_masked(values):
            raise ValueError(f"{name} may not contain masked or missing identifiers.")
    try:
        vector = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional categorical vector.") from exc
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a non-empty one-dimensional categorical vector.")
    kinds: set[str] = set()
    for value in vector:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"{name} contains an empty identifier.")
            kinds.add("string")
        elif isinstance(value, (bool, np.bool_)):
            kinds.add("boolean")
        elif isinstance(value, Integral):
            kinds.add("numeric")
        elif isinstance(value, Real):
            try:
                valid = np.isfinite(value) and (not integral or value == np.trunc(value))
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{name} numeric values must be finite" + (" integers." if integral else "."))
            kinds.add("numeric")
        else:
            raise ValueError(f"{name} must contain non-missing strings, booleans or finite numeric identifiers.")
    if len(kinds) > 1:
        raise ValueError(f"{name} must not mix string, boolean and numeric identifiers.")
    return vector.copy()


def validate_groups(values: Any, n: int, name: str) -> Any:
    """Validate homogeneous group IDs without merging strings and numbers."""
    n = _validate_count(n)
    if values is None:
        return None
    vector = _categorical_vector(values, name, integral=False)
    if len(vector) != n:
        raise ValueError(f"{name} must be aligned with every episode.")
    return vector


def _validate_labels(values: Any, name: str) -> np.ndarray:
    """Validate discrete labels before any mixed-type NumPy coercion."""
    return _categorical_vector(values, name, integral=True)


def validate_dataset(dataset: Any) -> None:
    """Validate label/schema alignment and external class/group boundaries."""
    labels, groups = {}, {}
    for role in ("train", "test"):
        X = getattr(dataset, "X_" + role)
        y = _validate_labels(getattr(dataset, "y_" + role), "y_" + role)
        meta = getattr(dataset, role + "_metadata")
        if len(X) != len(y):
            raise ValueError(f"Invalid external {role} observations/labels.")
        if not isinstance(meta, pd.DataFrame) or len(meta) != len(y):
            raise ValueError(f"Unaligned {role} metadata.")
        if "sample_id" in meta and (meta.sample_id.isna().any() or meta.sample_id.duplicated().any()):
            raise ValueError(f"{role} sample_id must be unique and non-missing.")
        labels[role] = y
        groups[role] = validate_groups(getattr(dataset, "groups_" + role, None), len(y), "groups_" + role)
    _validate_labels(list(labels["train"]) + list(labels["test"]), "combined external labels")
    if len(set(labels["train"])) < 2 or not set(labels["test"]) <= set(labels["train"]):
        raise ValueError("Training must contain every test class and at least two classes.")
    a, b = groups["train"], groups["test"]
    if (a is None) != (b is None):
        raise ValueError("Both train/test group vectors are required together.")
    if a is not None:
        _categorical_vector(list(a) + list(b), "combined external groups", integral=False)
        if set(a).intersection(set(b)):
            raise ValueError("External train/test group disjointness violated.")


def dataset_identity(dataset: Any) -> dict:
    """Fingerprint actual types in labels and metadata, without string coercion."""
    result = {}
    for role in ("train", "test"):
        raw = getattr(dataset, "X_" + role + "_raw", None)
        if raw is None:
            raw = getattr(dataset, "X_" + role)
        result["raw_" + role] = batch_hash(raw)
        result["prepared_" + role] = batch_hash(getattr(dataset, "X_" + role))
        result["labels_" + role] = array_hash(getattr(dataset, "y_" + role))
        result["groups_" + role] = array_hash(getattr(dataset, "groups_" + role, None))
        frame = getattr(dataset, role + "_metadata")
        result["metadata_" + role] = stable_hash({
            "columns": list(frame.columns), "dtypes": [str(dtype) for dtype in frame.dtypes],
            "rows": frame.to_numpy(dtype=object).tolist(),
        })
    return result


def cost_identity(cost: Any) -> str:
    """Fingerprint cost data and serialized callable state when present.

    Callable serialization is optional and environment-dependent; record package
    versions and implementation_identity alongside this hash. A missing optional
    serializer is an error, never a fallback to a function name or object repr.
    """
    payload = asdict(cost)
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            payload[key] = array_hash(value)
        elif callable(value):
            try:
                import cloudpickle
            except ImportError as exc:
                raise ImportError("Callable cost fingerprints require optional cloudpickle; core GRACE validation does not.") from exc
            payload[key] = {"serializer": "cloudpickle", "version": cloudpickle.__version__,
                            "sha256": hashlib.sha256(cloudpickle.dumps(value)).hexdigest()}
    return stable_hash(payload)


def implementation_identity() -> dict:
    """Identify this local GRACE implementation, not author authenticity."""
    root = Path(__file__).resolve().parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    return {"product": PRODUCT, "hash_schema": HASH_SCHEMA,
            "source_hash": stable_hash(files), "files": files}
