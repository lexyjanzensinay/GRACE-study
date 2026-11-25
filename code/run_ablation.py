"""Reproducible standalone GRACE ablations on synthetic or prepared episodes.

Run ``python run_ablation.py --config configs/synthetic_smoke.json --output results/smoke``.
This runner never downloads data and never selects a model using external-test labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grace_core import (GRACE, evaluate_grace, generate_synthetic_early_dataset,
                        multiclass_brier_score, top_label_ece, _clip_probabilities_for_logloss)
from grace_validation import array_hash, validate_groups, validate_temporal, utc_nanoseconds, _validate_labels

FEATURES = tuple(GRACE.THEORY_FEATURES)
VARIANTS: dict[str, dict[str, Any]] = {}
for _name, _family, _history in (
    ("GRACE-C", "conjunctive", "soft"),
    ("GRACE-LS", "learned", "soft"),
    ("GRACE-LH", "learned", "eligible"),
):
    _base = {"gated_acceptance_family": _family, "learned_history_mode": _history,
             "gated_acceptance_features": FEATURES}
    VARIANTS[_name] = _base
    VARIANTS[_name + "-NoCalibration"] = {**_base, "calibration_method": "none"}
    VARIANTS[_name + "-NoStability"] = {
        **_base, "gated_acceptance_features": tuple(x for x in FEATURES if x != "action_stability")}
    VARIANTS[_name + "-NoRisk"] = {
        **_base, "gated_acceptance_features": tuple(x for x in FEATURES if x not in {"risk_min", "risk_margin"})}
for _family_name in ("GRACE-C", "GRACE-LS", "GRACE-LH"):
    VARIANTS[_family_name + "-PmaxOnly"] = {
        **VARIANTS[_family_name], "gated_acceptance_features": ("pmax",)}

# Intentionally small declarative API. Unknown options fail rather than disappear.
TOP_KEYS = {"schema_version", "study_name", "data", "split", "model", "grace",
            "variants", "seeds", "alpha_delay", "bootstrap", "notes"}
GRACE_KEYS = {"decision_points", "calibration_method", "fit_mode", "crossfit_folds",
              "crossfit_inner_folds", "crossfit_purge", "crossfit_embargo", "calibration_size",
              "reliability_size", "gated_acceptance_size", "gated_acceptance_validation_fraction",
              "action_stability_window", "gated_acceptance_quantiles", "gated_acceptance_coordinate_passes",
              "learned_acceptance_thresholds", "learned_acceptance_crossfit_folds",
              "alpha_observation", "misclassification_cost", "cost_class_order"}
TIME_KEYS = ("episode_time", "information_start", "information_end")


@dataclass(frozen=True)
class Episodes:
    X: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    metadata: dict[str, np.ndarray]
    groups: np.ndarray | None
    provenance: dict[str, Any]


def json_ready(value: Any) -> Any:
    """Produce strict JSON; explicitly represent nonfinite schedule thresholds."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            if not isinstance(value,np.floating):
                raise TypeError(f"Unsupported NumPy scalar for JSON: {value.dtype}")
            converted = float(value)
            if np.isfinite(value) and (not math.isfinite(converted) or value.dtype.type(converted) != value):
                raise ValueError("Extended-precision number cannot be represented exactly as a JSON float; convert it explicitly before serialization.")
            scalar = converted
        return json_ready(scalar)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "positive_infinity" if value > 0 else "negative_infinity" if value < 0 else "not_a_number"
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _keys(value: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    """Read an explicit study configuration without accepting unknown fields."""
    config = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_json_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON constant: {value}")))
    _keys(config, TOP_KEYS, "configuration")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1.")
    if not isinstance(config.get("study_name"), str) or not config["study_name"].strip():
        raise ValueError("study_name must be non-empty.")
    _keys(config.get("data"), {"kind", "path", "n_samples", "length", "channels", "noise", "seed",
                               "feature_provenance", "label_provenance", "preparation_manifest"}, "data")
    if config["data"].get("kind") not in {"synthetic", "npz"}:
        raise ValueError("data.kind must be synthetic or npz.")
    if "notes" in config and (not isinstance(config["notes"], list) or not all(isinstance(x, str) for x in config["notes"])):
        raise ValueError("notes must be a list of strings.")
    if config["data"]["kind"] == "synthetic":
        incompatible = set(config["data"]) & {"path", "preparation_manifest", "feature_provenance", "label_provenance"}
        if incompatible:
            raise ValueError(f"Synthetic input has unused NPZ options: {sorted(incompatible)}")
        for key, default, minimum in (("n_samples",450,24),("length",60,8),("channels",2,1),("seed",20261003,0)):
            _integer(config["data"].get(key, default), "data."+key, minimum)
        if config["data"].get("seed",20261003) >= 2**32:
            raise ValueError("data.seed must be below 2**32.")
        noise = config["data"].get("noise", .65)
        if isinstance(noise,bool) or not isinstance(noise,(float,int)) or not math.isfinite(noise) or noise < 0:
            raise ValueError("data.noise must be finite, numeric and non-negative.")
    else:
        incompatible = set(config["data"]) & {"n_samples","length","channels","noise","seed"}
        if incompatible:
            raise ValueError(f"NPZ input has unused synthetic options: {sorted(incompatible)}")
        for key in ("path", "preparation_manifest"):
            if key in config["data"] and (not isinstance(config["data"][key],str) or not config["data"][key].strip()):
                raise ValueError(f"data.{key} must be a non-empty path string.")
    split = _keys(config.get("split"), {"strategy", "test_fraction", "test_start", "seed", "purge_rows", "embargo_rows"}, "split")
    if split.get("strategy") not in {"chronological", "stratified", "group"}:
        raise ValueError("split.strategy must be chronological, stratified, or group.")
    if ("test_fraction" in split) == ("test_start" in split):
        raise ValueError("Declare exactly one of split.test_fraction or split.test_start.")
    if "test_start" in split:
        if split["strategy"] != "chronological":
            raise ValueError("split.test_start is supported only with chronological splitting.")
        parse_test_start(split["test_start"])
    else:
        fraction = split.get("test_fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 < fraction < 1:
            raise ValueError("split.test_fraction must lie strictly in (0,1).")
    for key in ("seed", "purge_rows", "embargo_rows"):
        _integer(split.get(key, 0), "split." + key)
    if split.get("seed", 0) >= 2**32:
        raise ValueError("split.seed must be below 2**32.")
    if split["strategy"] != "chronological" and (split.get("purge_rows", 0) or split.get("embargo_rows", 0)):
        raise ValueError("Row purge/embargo requires chronological splitting.")
    _validate_model_settings(config.get("model"))
    _keys(config.get("grace"), GRACE_KEYS, "grace")
    if config["grace"].get("calibration_method", "temperature") not in {"temperature", "sigmoid", "isotonic", "none"}:
        raise ValueError("Invalid calibration_method.")
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants or not all(isinstance(x, str) and x in VARIANTS for x in variants):
        raise ValueError(f"variants must be a non-empty list selected from {sorted(VARIANTS)}")
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate variants are not allowed.")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list.")
    for seed in seeds:
        if _integer(seed, "model seed") >= 2**32:
            raise ValueError("Model seed must be below 2**32.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate model seeds are not independent replicates.")
    alphas = config.get("alpha_delay")
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("alpha_delay must be a non-empty list.")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in alphas):
        raise ValueError("alpha_delay entries must be finite and non-negative.")
    if len(alphas) != len(set(alphas)):
        raise ValueError("Duplicate alpha_delay settings are not allowed.")
    boot = _keys(config.get("bootstrap", {}), {"replicates", "mean_block_length", "seed", "confidence"}, "bootstrap")
    _integer(boot.get("replicates", 0), "bootstrap.replicates")
    _integer(boot.get("seed", 1729), "bootstrap.seed")
    mean_block = boot.get("mean_block_length", 1)
    if isinstance(mean_block, bool) or not isinstance(mean_block, (float, int)) or not math.isfinite(mean_block) or mean_block < 1:
        raise ValueError("bootstrap.mean_block_length must be finite and >=1.")
    confidence = boot.get("confidence", .95)
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 < confidence < 1:
        raise ValueError("bootstrap.confidence must lie strictly in (0,1).")
    if split["strategy"] == "stratified" and mean_block != 1:
        raise ValueError("Stratified independent episodes require bootstrap.mean_block_length=1.")
    if split["strategy"] == "group" and mean_block != 1:
        raise ValueError("Whole-group bootstrap does not use a stationary block length; set bootstrap.mean_block_length=1.")
    # Validate every family-dependent hyperparameter before loading data or writing output.
    for variant in config["variants"]:
        make_model(config, variant, config["seeds"][0], config["alpha_delay"][0])._validate_hyperparameters()
    return config


def validate_cost_contract(config: dict, classes: np.ndarray) -> np.ndarray:
    """Reject ambiguous class types before the core maps loss rows and columns."""
    declared = config["grace"].get("cost_class_order")
    if declared is None:
        if config["grace"].get("misclassification_cost") is not None:
            raise ValueError("Custom misclassification_cost requires explicit cost_class_order.")
        return classes.copy()
    order = _validate_labels(declared, "cost_class_order")
    _validate_labels(list(classes) + list(order), "dataset labels and cost_class_order")
    if len(order) != len(classes) or len(set(order)) != len(order) or set(order) != set(classes):
        raise ValueError("cost_class_order must uniquely list exactly every dataset and retained training class.")
    return order


def planned_runs(config: dict) -> list[dict[str, Any]]:
    """Enumerate the complete prespecified grid, independent of any results."""
    return [{"variant": variant, "seed": seed, "alpha_delay": alpha,
             "run_name": run_name(variant, seed, alpha)}
            for seed in config["seeds"] for alpha in config["alpha_delay"] for variant in config["variants"]]


def validate_episodes(X: Any, y: Any, sample_id: Any = None, metadata: dict | None = None,
                      groups: Any = None, provenance: dict | None = None) -> Episodes:
    """Validate the dense, fixed-length interchange contract before splitting."""
    for name, values in (("X", X), ("y", y), ("sample_id", sample_id)):
        if isinstance(values, np.ma.MaskedArray) and np.ma.is_masked(values):
            raise ValueError(f"Dense input {name} may not contain masked values; masks cannot be discarded.")
    raw = np.asarray(X)
    if raw.ndim != 3 or any(x < 1 for x in raw.shape) or raw.dtype.kind not in "fiu":
        raise ValueError("X must be a numeric (episodes, time, channels) array with non-empty axes.")
    X_array = np.asarray(raw, dtype=float)
    if not np.isfinite(X_array).all():
        raise ValueError("Dense NPZ input requires finite observed values; no missing values are imputed.")
    y_array = np.asarray(y)
    if y_array.ndim != 1 or len(y_array) != len(X_array) or y_array.dtype.kind not in "USiufb":
        raise ValueError("y must be aligned Unicode strings or finite integer-valued class labels; object arrays are prohibited.")
    if y_array.dtype.kind in "US":
        y_array = y_array.astype(str)
        if any(not x.strip() for x in y_array):
            raise ValueError("Empty class labels are invalid.")
    elif not np.isfinite(y_array).all() or not np.equal(y_array, np.floor(y_array)).all():
        raise ValueError("Numeric class labels must be finite integers.")
    else:
        limits = np.iinfo(np.int64)
        # Python int conversion avoids float rounding of int64.max to 2**63.
        if any(int(value) < limits.min or int(value) > limits.max for value in y_array):
            raise ValueError("Numeric class labels must fit int64 without overflow.")
        y_array = y_array.astype(np.int64)
    if len(np.unique(y_array)) < 2:
        raise ValueError("At least two outcome classes are required.")
    if sample_id is None:
        ids = np.asarray([f"episode_{i:08d}" for i in range(len(y_array))])
    else:
        ids = np.asarray(sample_id)
        if ids.ndim != 1 or len(ids) != len(y_array) or ids.dtype.kind not in "USiu":
            raise ValueError("sample_id must be an aligned non-missing string or integer vector.")
        ids = ids.astype(str)
    if any(not x.strip() for x in ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("sample_id must be unique and non-empty.")
    meta = {} if metadata is None else dict(metadata)
    if set(meta) and set(meta) != set(TIME_KEYS):
        raise ValueError("Supply all three temporal metadata vectors together.")
    if meta:
        validate_temporal(len(y_array), **meta)
    group_array = validate_groups(groups, len(y_array), "groups")
    return Episodes(X_array, y_array, ids, meta, group_array, provenance or {})


def load_episodes(config: dict, config_path: Path, data_override: Path | None = None) -> Episodes:
    """Read unpickled NPZ arrays or a labelled synthetic execution example."""
    data = config["data"]
    if data["kind"] == "synthetic":
        if data_override is not None:
            raise ValueError("--data is supported only for data.kind=npz.")
        noise = data.get("noise", .65)
        if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0:
            raise ValueError("data.noise must be finite, numeric and non-negative.")
        X, y = generate_synthetic_early_dataset(
            n_samples=_integer(data.get("n_samples", 450), "n_samples", 24),
            T=_integer(data.get("length", 60), "length", 8),
            n_channels=_integer(data.get("channels", 2), "channels", 1),
            noise=float(noise), random_state=_integer(data.get("seed", 20261003), "data.seed"))
        return validate_episodes(X, y.astype(str), provenance={
            "kind": "synthetic_execution_example", "generator": "generate_synthetic_early_dataset",
            "parameters": data, "not_empirical_market_evidence": True,
            "not_full_manuscript_AR_simulation": True})
    if not all(isinstance(data.get(x), str) and data[x].strip() for x in ("feature_provenance", "label_provenance")):
        raise ValueError("NPZ input requires explicit feature_provenance and label_provenance declarations.")
    if any(data[x].strip().upper().startswith("REQUIRED:") for x in ("feature_provenance", "label_provenance")):
        raise ValueError("Replace both REQUIRED provenance placeholders with the actual data preparation declarations.")
    supplied = (data_override if data_override is not None else Path(data.get("path", ""))).expanduser()
    if data_override is None and not data.get("path"):
        raise ValueError("Provide data.path or --data for NPZ input.")
    path = supplied.resolve() if data_override is not None or supplied.is_absolute() else (config_path.parent / supplied).resolve()
    with np.load(path, allow_pickle=False) as arrays:
        expected = {"X", "y", "sample_id", "groups", *TIME_KEYS}
        if not {"X", "y"} <= set(arrays.files) or set(arrays.files) - expected:
            raise ValueError(f"NPZ requires X,y and permits only {sorted(expected)}.")
        values = {key: arrays[key].copy() for key in arrays.files}
    provenance = {
        "kind": "prepared_npz", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feature_provenance": data["feature_provenance"], "label_provenance": data["label_provenance"],
        "provenance_declarations_verified_from_source": False}
    if "preparation_manifest" in data:
        declared_manifest = Path(data["preparation_manifest"]).expanduser()
        manifest_path = declared_manifest.resolve() if declared_manifest.is_absolute() else (config_path.parent / declared_manifest).resolve()
        prepared = json.loads(manifest_path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_json_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite preparation manifest: {value}")))
        if not isinstance(prepared, dict) or prepared.get("schema") != "grace-market-preparation-v1":
            raise ValueError("Preparation manifest must use the recognized grace-market-preparation-v1 schema.")
        if prepared.get("status") != "prepared":
            raise ValueError("Preparation manifest must have status='prepared'; incomplete or failed input is prohibited.")
        artifacts = prepared.get("artifacts_sha256")
        if not isinstance(artifacts,dict) or not artifacts or artifacts.get("episodes.npz") != provenance["sha256"]:
            raise ValueError("Preparation manifest must bind the exact input episodes.npz SHA256.")
        for name, digest in artifacts.items():
            if (not isinstance(name,str) or "\\" in name or ":" in name or Path(name).is_absolute()
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or not isinstance(digest,str) or len(digest)!=64):
                raise ValueError("Unsafe preparation artifact path or hash.")
            artifact = manifest_path.parent / name
            if artifact.is_symlink() or not artifact.is_file() or not artifact.resolve().is_relative_to(manifest_path.parent):
                raise ValueError("Preparation artifact is missing or outside its directory.")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Preparation artifact checksum mismatch: {name}")
        for key in ("feature_provenance", "label_provenance"):
            if prepared.get(key) != data[key]:
                raise ValueError(f"Declared {key} differs from the linked preparation manifest.")
        provenance["preparation_manifest"] = {"path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "all_listed_artifact_hashes_verified": True,
            "source_authenticity_verified": False, "contents": prepared}
    return validate_episodes(values["X"], values["y"], values.get("sample_id"),
        {x: values[x] for x in TIME_KEYS if x in values}, values.get("groups"), provenance)


def parse_test_start(value: Any) -> int:
    """Parse a prespecified timezone-aware calendar cutoff into UTC nanoseconds."""
    if not isinstance(value,str) or not value.strip():
        raise ValueError("split.test_start must be a timezone-aware ISO timestamp string.")
    try:
        timestamp=pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("Timestamp must be finite and timezone-aware.")
        return int(utc_nanoseconds([value], 1, "split.test_start")[0])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("split.test_start must be a valid timezone-aware ISO timestamp within nanosecond range.") from exc

def external_split(data: Episodes, options: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Freeze one test set and purge complete information intervals before all fits."""
    n = len(data.y)
    indices = np.arange(n)
    strategy = options["strategy"]
    if ("test_fraction" in options) == ("test_start" in options):
        raise ValueError("Declare exactly one of split.test_fraction or split.test_start.")
    test_fraction = float(options["test_fraction"]) if "test_fraction" in options else None
    if "test_start" in options and strategy != "chronological":
        raise ValueError("split.test_start requires chronological splitting.")
    seed = int(options.get("seed", 42))
    removed = np.array([], dtype=int)
    row_gap = int(options.get("purge_rows", 0)) + int(options.get("embargo_rows", 0))
    if data.groups is not None and strategy != "group":
        raise ValueError("Group-labelled input requires split.strategy=group; combined time-and-group protocols are unsupported.")
    if data.metadata and strategy != "chronological":
        raise ValueError("Temporal metadata requires chronological splitting; shuffling it is not permitted.")
    if strategy == "chronological":
        if not data.metadata:
            raise ValueError("Chronological input requires episode_time, information_start, information_end.")
        meta = validate_temporal(n, **data.metadata)
        cut = (int(np.searchsorted(meta["episode_time"],parse_test_start(options["test_start"]),side="left"))
               if "test_start" in options else n - int(math.ceil(n * test_fraction)))
        if cut < 1 or cut >= n:
            raise ValueError("The prespecified external cutoff leaves an empty training or test period.")
        test = indices[cut:]
        train = indices[:max(0, cut - row_gap)]
        boundary = np.min(meta["information_start"][test])
        keep = ((meta["information_end"][train] < boundary)
                & (meta["episode_time"][train] < np.min(meta["episode_time"][test])))
        train = train[keep]
        removed = np.setdiff1d(indices[:cut], train)
    elif strategy == "group":
        if data.groups is None:
            raise ValueError("groups are required for group splitting.")
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(indices, data.y, data.groups))
        train, test = np.sort(train), np.sort(test)
        if set(data.groups[train]) & set(data.groups[test]):
            raise RuntimeError("External group isolation failed.")
    elif strategy == "stratified":
        train, test = train_test_split(indices, test_size=test_fraction, random_state=seed, stratify=data.y)
        train, test = np.sort(train), np.sort(test)
    else:
        raise ValueError("Unsupported split.strategy.")
    if len(train) < 16 or not len(test):
        raise ValueError("External split/purging leaves insufficient training or test observations.")
    if len(np.unique(data.y[train])) < 2 or not set(np.unique(data.y[test])) <= set(np.unique(data.y[train])):
        raise ValueError("Training must contain every external-test class and at least two classes.")
    report = {"strategy": strategy, "options": options, "train_indices": train, "test_indices": test,
              "excluded_indices": removed, "train_sample_ids": data.sample_id[train],
              "test_sample_ids": data.sample_id[test], "n_train": len(train), "n_test": len(test),
              "same_split_for_every_variant_and_seed": True,
              "seed_changes_model_fit_not_external_split": True}
    if strategy == "chronological":
        report.update({"latest_training_information_end_ns": int(np.max(meta["information_end"][train])),
                       "earliest_test_information_start_ns": int(np.min(meta["information_start"][test])),
                       "first_test_episode_time_ns":int(np.min(meta["episode_time"][test])),
                       "last_test_episode_time_ns":int(np.max(meta["episode_time"][test])),
                       "test_boundary": "prespecified_calendar_cutoff" if "test_start" in options else "prespecified_fraction",
                       "embargo_semantics": "Additional pre-test rows withheld; forward-only protocol."})
    report["split_sha256"] = hash_json(report)
    return train, test, report


MODEL_KEYS = {"predictor_kind", "logistic_c", "gate_logistic_c", "max_iter", "class_weight",
              "hgb_max_leaf_nodes", "hgb_l2_regularization", "hgb_learning_rate", "hgb_max_iter"}
HGB_KEYS = {"hgb_max_leaf_nodes", "hgb_l2_regularization", "hgb_learning_rate", "hgb_max_iter"}


def _validate_legacy_model_settings(settings: Any) -> dict:
    """Strict pilot model schema; reject unused or unsupported options."""
    _keys(settings, MODEL_KEYS, "model")
    kind = settings.get("predictor_kind", "logistic")
    if not isinstance(kind, str) or kind not in {"logistic", "hist_gradient_boosting"}:
        raise ValueError("model.predictor_kind must be logistic or hist_gradient_boosting.")
    for key in ("logistic_c", "gate_logistic_c"):
        value = settings.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"model.{key} must be finite and positive.")
    # max_iter remains the logistic gate limit, and also the prefix limit for logistic.
    _integer(settings.get("max_iter", 3000), "model.max_iter", 1)
    weight = settings.get("class_weight")
    if weight is not None and (not isinstance(weight, str) or weight != "balanced"):
        raise ValueError("model.class_weight must be null or balanced.")
    if kind == "logistic":
        unused = HGB_KEYS.intersection(settings)
        if unused:
            raise ValueError(f"Histogram boosting options are unused by logistic: {sorted(unused)}")
    else:
        unused = {"logistic_c", "class_weight"}.intersection(settings)
        if unused:
            raise ValueError(f"Logistic prefix options are unused by histogram boosting: {sorted(unused)}")
        _integer(settings.get("hgb_max_leaf_nodes", 15), "model.hgb_max_leaf_nodes", 2)
        _integer(settings.get("hgb_max_iter", 200), "model.hgb_max_iter", 1)
        for key, default, strict_positive in (("hgb_learning_rate", 0.05, True),
                                               ("hgb_l2_regularization", 0.0, False)):
            value = settings.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or (value <= 0 if strict_positive else value < 0)):
                raise ValueError(f"model.{key} must be finite and {'positive' if strict_positive else 'non-negative'}.")
    return settings


def _validate_model_settings(settings: Any) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("model must be a dictionary")
    kind = settings.get("predictor_kind", "logistic")
    if kind not in {"random_forest", "svm_rbf"}:
        return _validate_legacy_model_settings(settings)
    common = {"predictor_kind", "gate_logistic_c", "max_iter", "class_weight"}
    specific = ({"rf_n_estimators", "rf_max_depth", "rf_min_samples_leaf"}
                if kind == "random_forest" else {"svm_c"})
    _keys(settings, common | specific, "model")
    _validate_legacy_model_settings({k: v for k, v in settings.items()
                                    if k in common and k != "predictor_kind"})
    if kind == "random_forest":
        for key, default in (("rf_n_estimators", 300), ("rf_max_depth", 8), ("rf_min_samples_leaf", 10)):
            _integer(settings.get(key, default), "model." + key, 1)
    else:
        v = settings.get("svm_c", 1.0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError("model.svm_c must be finite and positive")
    return settings


def make_model(config: dict, variant: str, seed: int, alpha: float) -> GRACE:
    """Build one prespecified variant; fitted scaling remains within each model fit."""
    options = dict(config["grace"])
    settings = _validate_model_settings(config["model"])
    max_iter = int(settings.get("max_iter", 3000))
    if settings.get("predictor_kind", "logistic") == "logistic":
        prefix = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(
            C=float(settings.get("logistic_c", 1.0)), max_iter=max_iter, random_state=seed,
            class_weight=settings.get("class_weight")))])
    elif settings.get("predictor_kind") == "random_forest":
        if options.get("calibration_method") != "sigmoid":
            raise ValueError("RF study requires explicit temporal sigmoid calibration")
        prefix = RandomForestClassifier(
            n_estimators=int(settings.get("rf_n_estimators", 300)),
            max_depth=int(settings.get("rf_max_depth", 8)),
            min_samples_leaf=int(settings.get("rf_min_samples_leaf", 10)),
            max_features="sqrt", class_weight=settings.get("class_weight"),
            random_state=seed, n_jobs=2)
    elif settings.get("predictor_kind") == "svm_rbf":
        if options.get("calibration_method") != "sigmoid":
            raise ValueError("SVM study requires explicit temporal sigmoid calibration")
        prefix = Pipeline([("scale", StandardScaler()), ("svm", SVC(
            C=float(settings.get("svm_c", 1.0)), kernel="rbf", gamma="scale",
            class_weight=settings.get("class_weight"), probability=False,
            decision_function_shape="ovr", random_state=seed, cache_size=256))])
    else:
        # No random validation split: all calibration, gate selection and validation
        # remain in GRACE's explicit development roles. Trees need no scaling.
        prefix = HistGradientBoostingClassifier(
            max_leaf_nodes=int(settings.get("hgb_max_leaf_nodes", 15)),
            l2_regularization=float(settings.get("hgb_l2_regularization", 0.0)),
            learning_rate=float(settings.get("hgb_learning_rate", 0.05)),
            max_iter=int(settings.get("hgb_max_iter", 200)),
            early_stopping=False, random_state=seed)
    gate = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(
        C=float(settings.get("gate_logistic_c", 1.0)), max_iter=max_iter,
        random_state=(seed + 401) % (2**32)))])
    options.setdefault("calibration_method", "temperature")
    options.update(VARIANTS[variant])
    options.update({"prefix_model": prefix, "gated_acceptance_model": gate,
                    "split_strategy": config["split"]["strategy"], "random_state": seed,
                    "alpha_delay": float(alpha), "reliability_mode": "fixed",
                    "auto_safe_mode_on_shift": False, "force_terminal_full": True})
    if options.get("misclassification_cost") is not None:
        options["misclassification_cost"] = np.asarray(options["misclassification_cost"], dtype=float)
    return GRACE(**options)


def bootstrap_indices(n: int, replicates: int, seed: int, strategy: str,
                      mean_block_length: float = 1, groups: np.ndarray | None = None) -> list[np.ndarray]:
    """Generate shared paired resamples: stationary time blocks or whole groups."""
    if n < 1 or replicates < 0 or mean_block_length < 1:
        raise ValueError("Invalid bootstrap dimensions or mean block length.")
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    if strategy == "group":
        if groups is None or len(groups) != n:
            raise ValueError("Group bootstrap requires aligned test group labels.")
        levels = np.unique(groups)
        members = [np.flatnonzero(groups == level) for level in levels]
        for _ in range(replicates):
            selected = rng.integers(0, len(levels), len(levels))
            result.append(np.concatenate([members[k] for k in selected]))
    elif strategy in {"chronological", "stratified"}:
        for _ in range(replicates):
            idx = np.empty(n, dtype=int)
            idx[0] = int(rng.integers(n))
            for j in range(1, n):
                idx[j] = int(rng.integers(n)) if rng.random() < 1.0 / mean_block_length else (idx[j - 1] + 1) % n
            result.append(idx)
    else:
        raise ValueError("Unknown bootstrap strategy.")
    return result


def paired_intervals(results: dict[str, dict[str, np.ndarray]], samples: list[np.ndarray],
                     confidence: float) -> list[dict[str, Any]]:
    """Percentile intervals for paired means conditional on frozen fitted models."""
    if "GRACE-C" not in results or not samples:
        return []
    alpha = (1.0 - confidence) / 2.0
    reports: list[dict[str, Any]] = []
    for variant, values in results.items():
        if variant == "GRACE-C":
            continue
        for metric in ("total_cost", "decision_fraction", "correct_action"):
            difference = values[metric] - results["GRACE-C"][metric]
            boot = np.asarray([float(np.mean(difference[index])) for index in samples])
            low, high = np.quantile(boot, [alpha, 1-alpha])
            reports.append({"variant": variant, "reference": "GRACE-C", "metric": metric,
                            "difference_variant_minus_reference": float(np.mean(difference)),
                            "ci_low": float(low), "ci_high": float(high), "confidence": confidence,
                            "scope": "conditional_on_frozen_models_and_external_test_set",
                            "full_training_uncertainty_included": False})
    return reports


def fixed_prefix_diagnostics(model: GRACE, probabilities: np.ndarray, y: np.ndarray,
                             variant: str, seed: int, alpha: float) -> list[dict[str, Any]]:
    """Score every prespecified fixed prefix on the same frozen probability paths.

    This is external evaluation only: no prefix is selected and no estimator or
    gate is refitted. The terminal classifier is the row with terminal=True.
    """
    encoded = model.label_encoder_.transform(y)
    reports = []
    for position, tau in enumerate(model.tau_grid_):
        p = probabilities[:, position]
        fraction = float(tau / model.T_)
        actions = np.asarray([model.cost_model_.bayes_action(row, tau_fraction=fraction) for row in p])
        costs = np.asarray([model.cost_model_.total_cost(int(truth),int(action),tau_fraction=fraction)
                            for truth,action in zip(encoded,actions)])
        reports.append({"variant":variant,"seed":seed,"alpha_delay":alpha,"tau":int(tau),
            "terminal":position == len(model.tau_grid_)-1,"n_test":len(y),"decision_fraction":fraction,
            "accuracy":float(np.mean(encoded==actions)),
            "macro_f1":float(f1_score(encoded,actions,labels=np.arange(model.n_classes_),average="macro",zero_division=0)),
            "balanced_accuracy":float(np.mean([np.mean(actions[encoded==label]==label) for label in np.unique(encoded)])),
            "avg_total_cost":float(np.mean(costs)),
            "brier_score":multiclass_brier_score(encoded,p),
            "log_loss":float(log_loss(encoded,_clip_probabilities_for_logloss(p),labels=np.arange(model.n_classes_))),
            "top_label_ece":top_label_ece(encoded,p),
            "probability_argmax_accuracy":float(np.mean(np.argmax(p,axis=1)==encoded)),
            "scope":"fixed-prefix comparator on the same frozen probability paths; no test-based prefix selection"})
    return reports

def run_name(variant: str, seed: int, alpha: float) -> str:
    """Use round-trip float text so distinct delay settings cannot collide."""
    return f"{variant}_seed-{seed}_alpha-{repr(float(alpha))}"


def save_predictions(path: Path, *, sample_id: np.ndarray, labels: np.ndarray,
                     times: np.ndarray, probabilities: np.ndarray,
                     prefix_probabilities: np.ndarray, classes: np.ndarray,
                     decision_points: np.ndarray,
                     temporal_metadata: dict[str, np.ndarray] | None = None) -> None:
    """Preserve numeric versus string class identity in the unpickled archive."""
    labels, classes = np.asarray(labels), np.asarray(classes)
    np.savez_compressed(path, sample_id=sample_id,
        predicted_label=labels.astype(str) if labels.dtype.kind == "O" else labels,
        stopping_time=times, probability_at_stop=probabilities,
        probability_by_prefix=prefix_probabilities,
        classes=classes.astype(str) if classes.dtype.kind == "O" else classes,
        decision_points=decision_points, **(temporal_metadata or {}))


def gate_feasibility(model: GRACE) -> dict[str, Any]:
    """Describe fitted training evidence, without selecting a policy on test data."""
    schedule = model.gated_acceptance_schedule_
    enabled = np.asarray(schedule.enabled_by_tau, dtype=bool)
    learned = model.selected_gated_acceptance_family_ == "learned"
    summary = getattr(model, "learned_acceptance_oof_summary_", {})
    scores = np.asarray(summary.get("finite_oof_scores_by_prefix", []), dtype=int)
    no_oof = learned and (not len(scores) or not np.any(scores[:-1] > 0))
    warnings = []
    if no_oof:
        warnings.append("No independent nonterminal learned-gate scores survived; terminal fallback is execution evidence, not an informative learned early-stopping comparison.")
    elif not np.any(enabled[:-1]):
        warnings.append("No nonterminal gate is enabled after development tuning; report the terminal-only policy explicitly.")
    return {"source":"frozen_training_artifacts_only", "family":model.selected_gated_acceptance_family_,
            "gate_development_rows":len(model.split_indices_.get("gated_acceptance_development", [])),
            "finite_oof_scores_by_prefix":scores.tolist() if learned else None,
            "enabled_nonterminal_prefixes":int(np.count_nonzero(enabled[:-1])),
            "learned_gate_has_independent_nonterminal_scores":None if not learned else not no_oof,
            "informative_learned_gate_comparison":not no_oof, "warnings":warnings}


def run_study(config_path: Path, output: Path, data_override: Path | None = None) -> dict:
    """Fit the complete fixed grid and export computational evidence."""
    _study_begin = time.perf_counter()
    _timing_runs = []
    config_path = config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    config = load_config(config_path)
    data = load_episodes(config, config_path, data_override)
    train, test, split = external_split(data, config["split"])
    classes = np.unique(data.y[train])
    if set(classes) != set(np.unique(data.y)):
        raise ValueError("Retained training data must contain every dataset class; no class is removed or relabelled automatically.")
    validate_cost_contract(config, classes)
    for variant in config["variants"]:
        validator = make_model(config, variant, config["seeds"][0], config["alpha_delay"][0])
        validator.classes_, validator.n_classes_ = classes, len(classes)
        validator._validate_cost_configuration()
        validator._resolve_tau_grid(data.X.shape[1])
    planned = planned_runs(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output}. Choose a new path to preserve previous evidence.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_resolved.json", config)
    write_json(output / "split.json", split)
    software = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "cloudpickle"):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = None
    implementation = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in ("grace_core.py", "grace_validation.py", "run_ablation.py")}
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "running", "study_name": config["study_name"],
        "config_sha256": hash_json(config), "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "data_provenance": data.provenance, "array_hashes": {"X": array_hash(data.X), "y": array_hash(data.y),
        "sample_id": array_hash(data.sample_id), "groups": array_hash(data.groups),
        **{key: array_hash(value) for key, value in data.metadata.items()}},
        "shape": data.X.shape, "implementation_sha256": implementation, "software": software,
        "python": sys.version, "platform": platform.platform(), "split_sha256": split["split_sha256"],
        "external_test_selection": "none: all configured variants/seeds/alpha values are reported",
        "convergence_warning_policy": "fail_run_before_prediction_or_metric_export",
        "model_seed_repetitions_are_not_independent_data_replicates": True,
        "preprocessing": ("13 causal within-prefix summaries per channel; "
                          + ("unscaled histogram gradient boosting prefix classifier; "
                             if config["model"].get("predictor_kind", "logistic") == "hist_gradient_boosting"
                             else "StandardScaler fitted inside each prefix classifier fit; ")
                          + "StandardScaler fitted inside each logistic gate fit"),
        "predictor_kind": config["model"].get("predictor_kind", "logistic"),
        "cost_orientation": "rows=actions, columns=true classes in cost_class_order",
        "variant_specifications": {name: VARIANTS[name] for name in config["variants"]},
        "planned_runs": planned, "planned_fit_count": len(planned),
        "metric_contract_version":2, "gate_feasibility_by_run":[],
        "fit_count_scope": "Top-level GRACE fits, excluding internal classifier/calibrator/gate fits.",
        "completed_runs": [], "notes": config.get("notes", [])}
    write_json(output / "manifest.json", manifest)
    boot_cfg = config.get("bootstrap", {})
    resamples = bootstrap_indices(len(test), int(boot_cfg.get("replicates", 0)), int(boot_cfg.get("seed", 1729)),
                                  config["split"]["strategy"], float(boot_cfg.get("mean_block_length", 1)),
                                  None if data.groups is None else data.groups[test])
    write_json(output / "bootstrap_design.json", {"strategy": config["split"]["strategy"],
               "method": "whole_group" if data.groups is not None else "stationary_circular" if config["split"]["strategy"] == "chronological" else "iid",
               "parameters": boot_cfg, "paired_index_hashes": [array_hash(x) for x in resamples],
               "shared_across_variants": True, "seeds_analyzed_separately": True,
               "interval_scope": "conditional fixed-model prediction uncertainty; training not repeated inside bootstrap"})
    all_metrics: list[dict] = []
    fixed_baselines: list[dict] = []
    intervals: list[dict] = []
    common_by_seed: dict[tuple, tuple[dict, np.ndarray, str]] = {}
    _preparation_end = time.perf_counter()
    try:
        for seed in config["seeds"]:
            for alpha in config["alpha_delay"]:
                paired: dict[str, dict[str, np.ndarray]] = {}
                for variant in config["variants"]:
                    name = run_name(variant, seed, alpha)
                    manifest["active_run"] = name
                    write_json(output / "manifest.json", manifest)
                    print(f"Fitting {name}", flush=True)
                    run_dir = output / name
                    run_dir.mkdir()
                    model = make_model(config, variant, seed, alpha)
                    fit_kwargs = {key: values[train] for key, values in data.metadata.items()}
                    if data.groups is not None:
                        fit_kwargs["groups"] = data.groups[train]
                    started = time.perf_counter()
                    # An optimizer warning is not a successful scientific fit.
                    # Catch it before predictions or publication metrics exist,
                    # and leave an explicit failed manifest for diagnosis.
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", ConvergenceWarning)
                            model.fit(data.X[train], data.y[train], **fit_kwargs)
                    except ConvergenceWarning as exc:
                        raise RuntimeError(
                            "An estimator failed to converge. Review training-only numerical diagnostics "
                            "and the prespecified max_iter setting before rerunning in a new output directory."
                        ) from exc
                    # Persist only the prespecified control for paired gate-only evaluation.
                    if (config["model"].get("predictor_kind") == "logistic"
                        and config["model"].get("logistic_c") == 0.01
                        and config["model"].get("class_weight") == "balanced"
                        and model.calibration_method == "sigmoid"):
                        import pickle
                        with (run_dir / "frozen_control_model.pkl").open("wb") as frozen_handle:
                            pickle.dump(model, frozen_handle, protocol=5)
                    fit_seconds = time.perf_counter() - started
                    _evaluation_begin = time.perf_counter()
                    feasibility = gate_feasibility(model)
                    manifest["gate_feasibility_by_run"].append({"run_name":name, **feasibility})
                    for warning in feasibility["warnings"]:
                        print(f"Study warning [{name}]: {warning}", file=sys.stderr, flush=True)
                    role_indices = {key: np.asarray(value).tolist() for key, value in model.split_indices_.items()}
                    prefix_probabilities = np.stack([model.predict_proba_at_tau(data.X[test], int(tau)) for tau in model.tau_grid_], axis=1)
                    # Paired raw/calibrated predictions use the same fitted estimators and preprocessing.
                    original_calibrators = model.calibrators_
                    try:
                        model.calibrators_ = model.base_models_
                        raw_probabilities = model._probability_grid(data.X[test])
                    finally:
                        model.calibrators_ = original_calibrators
                    argmax_mismatches = int(np.sum(np.argmax(raw_probabilities, axis=2) != np.argmax(prefix_probabilities, axis=2)))
                    if model.calibration_method == "temperature" and argmax_mismatches:
                        raise RuntimeError("Positive-temperature argmax parity failed.")
                    np.savez_compressed(run_dir / "raw_calibrated_pair.npz",
                        raw=raw_probabilities, calibrated=prefix_probabilities,
                        sample_id=data.sample_id[test], classes=model.classes_, tau=model.tau_grid_)
                    prefix_invariance = []
                    for position, tau in enumerate(model.tau_grid_[:-1]):
                        probe = np.array(data.X[test[:2]], copy=True)
                        reference = model.predict_proba_at_tau(probe, int(tau))
                        probe[:, int(tau):, :] = 12345.0
                        altered = model.predict_proba_at_tau(probe, int(tau))
                        delta = float(np.max(np.abs(reference - altered)))
                        if delta != 0.0:
                            raise RuntimeError("Future-suffix invariance failed.")
                        prefix_invariance.append({"tau": int(tau), "max_difference": delta})
                    write_json(run_dir / "paired_calibration_audit.json", {
                        "same_fitted_base": True, "argmax_mismatches": argmax_mismatches,
                        "future_suffix_invariance": prefix_invariance,
                        "interpretation": "Sigmoid may change argmax. Temperature must preserve it."})
                    shared_key = (seed, model.calibration_method)
                    if shared_key in common_by_seed:
                        ref_roles, ref_probabilities, ref_name = common_by_seed[shared_key]
                        prefix_max_difference = float(np.max(np.abs(prefix_probabilities - ref_probabilities)))
                        if role_indices != ref_roles or not np.allclose(prefix_probabilities, ref_probabilities, rtol=0, atol=1e-12):
                            raise RuntimeError("Controlled ablation failed: role indices or prefix probabilities differ for shared classifier/calibration settings.")
                    else:
                        common_by_seed[shared_key] = (role_indices, prefix_probabilities.copy(), name)
                        ref_name = name
                        prefix_max_difference = 0.0
                    # Predictions are persisted before external labels enter metric evaluation.
                    labels, times, probabilities, traces = model.predict_early_proba(data.X[test])
                    save_predictions(run_dir / "predictions.npz", sample_id=data.sample_id[test],
                                     labels=labels, times=times, probabilities=probabilities,
                                     prefix_probabilities=prefix_probabilities, classes=model.classes_,
                                     decision_points=model.tau_grid_,
                                     temporal_metadata={key: value[test] for key, value in data.metadata.items()})
                    write_json(run_dir / "decision_traces.json", traces)
                    metric = evaluate_grace(model, data.X[test], data.y[test])
                    baseline_rows = fixed_prefix_diagnostics(model,prefix_probabilities,data.y[test],variant,seed,alpha)
                    write_json(run_dir / "fixed_prefix_baselines.json", baseline_rows)
                    fixed_baselines.extend(baseline_rows)
                    encoded_true = model.label_encoder_.transform(data.y[test])
                    encoded_action = model.label_encoder_.transform(labels)
                    argmax_labels = model.classes_[np.argmax(probabilities, axis=1)]
                    fractions = np.asarray([trace["tau_fraction"] for trace in traces])
                    costs = np.asarray([model.cost_model_.total_cost(int(y), int(a), tau_fraction=float(f))
                                        for y, a, f in zip(encoded_true, encoded_action, fractions)])
                    paired[variant] = {"total_cost": costs, "decision_fraction": fractions,
                                       "correct_action": (labels == data.y[test]).astype(float)}
                    with (run_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["sample_id", "true_label", "bayes_action", "probability_argmax_label", "tau", "decision_fraction", "total_cost", "stop_reason"])
                        writer.writerows(zip(data.sample_id[test], data.y[test], labels, argmax_labels, times, fractions, costs, [x["stop_reason"] for x in traces]))
                    metric.update({"variant": variant, "seed": seed, "alpha_delay": alpha,
                                   "accuracy_is_bayes_action_accuracy": True,
                                   "probability_argmax_accuracy_at_stop": float(np.mean(argmax_labels == data.y[test])),
                                   "fit_seconds": fit_seconds, "prefix_probability_sha256": array_hash(prefix_probabilities),
                                   "gate_feasibility":feasibility,
                                   "shared_classifier_reference": ref_name,
                                   "shared_prefix_max_absolute_difference": prefix_max_difference,
                                   "shared_prefix_check_absolute_tolerance": 1e-12})
                    write_json(run_dir / "metrics.json", metric)
                    write_json(run_dir / "gate_schedule.json", model.gated_acceptance_schedule_.as_dict())
                    write_json(run_dir / "fit_audit.json", {
                        "split_summary": model.split_summary_, "training_local_role_indices": role_indices,
                        "training_original_role_indices": {key: train[np.asarray(value, dtype=int)] for key, value in role_indices.items()},
                        "temporal_split_audit": getattr(model, "temporal_split_audit_", []),
                        "crossfit_splits": getattr(model, "crossfit_splits_", []),
                        "calibration_inner_splits": getattr(model, "calibration_inner_splits_", []),
                        "final_calibration_split": getattr(model, "final_calibration_split_", {}),
                        "calibration_index_scope": getattr(model, "calibration_split_index_scope_", None),
                        "predictor_development_indices": getattr(model, "predictor_development_indices_", []),
                        "learned_acceptance_oof_summary": getattr(model, "learned_acceptance_oof_summary_", {}),
                        "gated_acceptance_candidate_summary": getattr(model, "gated_acceptance_candidate_summary_", {}),
                        "final_calibration_by_prefix": [{"tau": int(tau),
                            **{key: getattr(calibrator, key, None) for key in ("temperature_", "calibration_log_loss_before_",
                              "calibration_log_loss_after_", "optimization_status_", "n_calibration_rows_",
                              "effective_calibration_scheme_")}} for tau, calibrator in zip(model.tau_grid_, model.calibrators_)],
                        "gate_variant": VARIANTS[variant], "base_calibration_method": model.calibration_method,
                        "all_role_indices_refer_only_to_external_training": True})
                    _timing_runs.append({'run_name':name, 'fit_seconds':fit_seconds,
                                         **model.stage_timing_,
                                         'evaluation_output_seconds':time.perf_counter()-_evaluation_begin})
                    all_metrics.append(metric)
                    manifest["completed_runs"].append(name)
                    write_json(output / "manifest.json", manifest)
                for row in paired_intervals(paired, resamples, float(boot_cfg.get("confidence", .95))):
                    intervals.append({"seed": seed, "alpha_delay": alpha, **row})
        write_json(output / "metrics.json", all_metrics)
        write_json(output / "fixed_prefix_baselines.json", fixed_baselines)
        write_json(output / "paired_intervals.json", intervals)
        columns = ["variant", "seed", "alpha_delay", "n_test", "accuracy", "probability_argmax_accuracy_at_stop", "macro_f1", "balanced_accuracy",
                   "decision_fraction", "earliness_gain", "avg_total_cost", "avg_misclassification_cost", "avg_delay_penalty",
                   "brier_score_at_stop", "log_loss_at_stop", "top_label_ece_at_stop", "forced_terminal_rate", "fit_seconds", "prefix_probability_sha256"]
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metrics)
        if manifest["completed_runs"] != [row["run_name"] for row in planned] or len(all_metrics) != len(planned):
            raise RuntimeError("The completed results do not cover the complete prespecified grid.")
        manifest.pop("active_run", None)
        manifest["status"] = "completed"
        manifest["all_shared_prefix_probability_checks_passed"] = True
        manifest["all_learned_comparisons_have_independent_scores"] = all(
            row["informative_learned_gate_comparison"] for row in manifest["gate_feasibility_by_run"])
        write_json(output / "manifest.json", manifest)
        _timing_end = time.perf_counter()
        write_json(output / 'timing_stages.json', {'data_preparation_shared_seconds':_preparation_end-_study_begin,
                   'runs':_timing_runs, 'runner_elapsed_before_hash_export_seconds':_timing_end-_study_begin,
                   'unallocated_runner_seconds':(_timing_end-_study_begin)-(_preparation_end-_study_begin)-sum(x['fit_seconds']+x['evaluation_output_seconds'] for x in _timing_runs),
                   'definitions':{'predictor_training_calibration_seconds':'Predictor routine includes prefix feature extraction, crossfit training, calibration and OOF prediction',
                   'reliability_gate_seconds':'Post-predictor reliability fitting, gate features, gate fitting and validation, schedule finalization',
                   'evaluation_output_seconds':'Post-fit probabilities, evaluation, predictions and per-run result exports; final shared exports are unallocated runner overhead',
                   'total_end_to_last_output':'Use parent subprocess wall; includes interpreter imports and final timing/hash exports'}})
        file_hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS.json"}
        write_json(output / "SHA256SUMS.json", file_hashes)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "manifest.json", manifest)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Explicit JSON study configuration.")
    parser.add_argument("--data", type=Path, help="Override NPZ path, relative to the current working directory.")
    parser.add_argument("--output", type=Path, required=True, help="New or empty evidence directory; never overwritten.")
    args = parser.parse_args()
    run_study(args.config, args.output, args.data)
    print(f"Completed. Evidence: {args.output.resolve()}")


if __name__ == "__main__":
    main()
