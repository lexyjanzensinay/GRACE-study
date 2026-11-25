"""Independent GRACE core algorithms.

GRACE denotes Gated Reliability and Cost-Aware Calibration for Early Decisions.
The estimator implements current-prefix early decision making through five aligned
components: prefix prediction, probability calibration, reliability assessment,
gated acceptance, and cost-aware Bayes action selection.  Temporal validation,
purging, embargo, evaluation, and monitoring are handled without using future
information in an individual decision.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import warnings
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
try:
    from .grace_validation import validate_temporal, validate_groups, _validate_labels
except ImportError:
    from grace_validation import validate_temporal, validate_groups, _validate_labels
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import ks_2samp

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - compatibility path for older releases
    FrozenEstimator = None  # type: ignore[assignment,misc]




class CrossfitFeasibilityError(ValueError):
    """Training-design infeasibility with JSON-ready, training-only diagnostics."""

    def __init__(self, message: str, diagnostics: Dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = copy.deepcopy(diagnostics)


class _LogisticTemperatureCalibrator(ClassifierMixin, BaseEstimator):
    """Positive scalar temperature fitted to held-out compatible classifier logits.

    LogisticRegression and HistGradientBoostingClassifier, optionally as the
    final step of a Pipeline, are supported. The historical class name is kept
    for compatibility. Decision margins from arbitrary classifiers are not
    interpreted as multinomial logits. The bounded search uses log(T) in [-6, 6] and retains
    T=1 whenever optimization fails or does not improve held-out log loss.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def _logits(self, X: np.ndarray) -> np.ndarray:
        final = self.estimator.steps[-1][1] if isinstance(self.estimator, Pipeline) else self.estimator
        if not isinstance(final, (LogisticRegression, HistGradientBoostingClassifier)):
            raise TypeError("Temperature calibration requires LogisticRegression or HistGradientBoostingClassifier, optionally as the final Pipeline step.")
        classes = np.asarray(self.estimator.classes_)
        if len(classes) < 2:
            raise ValueError("Temperature calibration requires at least two fitted classes.")
        if isinstance(final, LogisticRegression) and len(classes) > 2 and (getattr(final, "multi_class", None) == "ovr" or final.solver == "liblinear"):
            raise ValueError("Multiclass temperature calibration requires multinomial logistic logits.")
        scores = np.asarray(self.estimator.decision_function(X), dtype=float)
        if len(classes) == 2 and scores.ndim == 1:
            logits = np.column_stack((np.zeros_like(scores), scores))
        else:
            logits = scores
        if logits.shape != (len(X), len(classes)) or not np.isfinite(logits).all():
            raise ValueError("Temperature logits must be finite and aligned with rows and classes.")
        if not np.allclose(softmax(logits, axis=1), self.estimator.predict_proba(X), rtol=1e-8, atol=1e-10):
            raise ValueError("Classifier probabilities do not agree with the supported binary/multinomial logits.")
        return logits

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticTemperatureCalibrator":
        logits = self._logits(X)
        self.classes_ = np.asarray(self.estimator.classes_).copy()
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(logits) or set(np.unique(labels)) != set(self.classes_):
            raise ValueError("Temperature calibration rows must be aligned and contain every fitted class.")
        columns = np.asarray([int(np.flatnonzero(self.classes_ == label)[0]) for label in labels])

        def objective(log_temperature: float) -> float:
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = logits / np.exp(log_temperature)
                losses = logsumexp(scaled, axis=1) - scaled[np.arange(len(columns)), columns]
            return float(np.mean(losses)) if np.isfinite(losses).all() else float("inf")

        identity_loss = objective(0.0)
        if not np.isfinite(identity_loss):
            raise ValueError("Calibration log loss is not finite at temperature one.")
        optimum = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-8})
        improves = bool(optimum.success and np.isfinite(optimum.fun) and optimum.fun < identity_loss)
        self.temperature_ = float(np.exp(optimum.x)) if improves else 1.0
        self.calibration_log_loss_before_ = identity_loss
        self.calibration_log_loss_after_ = objective(float(np.log(self.temperature_)))
        self.temperature_log_bounds_ = (-6.0, 6.0)
        self.optimization_status_ = "improved" if improves else "identity_retained"
        self.n_calibration_rows_ = int(len(labels))
        final = self.estimator.steps[-1][1] if isinstance(self.estimator, Pipeline) else self.estimator
        self.effective_calibration_scheme_ = (
            "explicit_hist_gradient_boosting_temperature_holdout"
            if isinstance(final, HistGradientBoostingClassifier)
            else "explicit_logistic_temperature_holdout")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["classes_", "temperature_"])
        logits = self._logits(X)
        if not np.array_equal(self.estimator.classes_, self.classes_):
            raise RuntimeError("Estimator class order changed after temperature calibration.")
        return softmax(logits / self.temperature_, axis=1)


__all__ = [
    "CostModel",
    "GatedAcceptanceSchedule",
    "GRACE",
    "MaskedSeriesBatch",
    "ReliabilityWeights",
    "default_prefix_transformer",
    "evaluate_calibration_by_prefix",
    "evaluate_grace",
    "generate_synthetic_early_dataset",
    "multiclass_brier_score",
    "reliability_features",
    "reliability_score",
    "top_label_ece",
]


@dataclass
class MaskedSeriesBatch:
    """Padded time-series values together with explicit observation metadata.

    Parameters
    ----------
    values:
        Numeric array with shape ``(n_samples, max_length, n_channels)``.
        Values at unobserved positions are ignored and internally replaced by
        zero to prevent accidental propagation of NaNs.
    observation_mask:
        Boolean array with the same shape as ``values``.  ``True`` marks an
        observed value.  Channel-wise missingness and padded tails are allowed.
    timestamps:
        Optional array with shape ``(n_samples, max_length)``.  Timestamps must
        be finite and strictly increasing at observed time points.  Integer
        positions are used when omitted.
    channel_names:
        Optional ordered channel schema.  Prediction data may omit trailing
        channels, but it may not introduce channels beyond the fitted schema.
    original_ndim:
        Retains whether a regular input was originally two- or three-
        dimensional so legacy custom transformers can receive a familiar view.
    """

    values: np.ndarray
    observation_mask: np.ndarray
    timestamps: Optional[np.ndarray] = None
    channel_names: Optional[Tuple[str, ...]] = None
    original_ndim: int = 3

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        mask = np.asarray(self.observation_mask, dtype=bool)
        if values.ndim != 3 or mask.shape != values.shape:
            raise ValueError(
                "values and observation_mask must have identical shape (n,T,d)."
            )
        n, T, d = values.shape
        if n < 1 or T < 1 or d < 1:
            raise ValueError("A batch needs at least one sample, time point, and channel.")
        if np.any(mask & ~np.isfinite(values)):
            raise ValueError("Observed values must be finite.")
        values = np.where(mask, values, 0.0)

        if self.timestamps is None:
            timestamps = np.tile(np.arange(T, dtype=float), (n, 1))
        else:
            timestamps = np.asarray(self.timestamps, dtype=float)
            if timestamps.shape != (n, T):
                raise ValueError("timestamps must have shape (n,T).")
        time_mask = np.any(mask, axis=2)
        if np.any(~np.any(time_mask, axis=1)):
            raise ValueError("Every sample must contain at least one observed time point.")
        for i in range(n):
            observed_times = timestamps[i, time_mask[i]]
            if not np.all(np.isfinite(observed_times)):
                raise ValueError("Observed timestamps must be finite.")
            if len(observed_times) > 1 and np.any(np.diff(observed_times) <= 0.0):
                raise ValueError("Observed timestamps must be strictly increasing.")

        if self.channel_names is not None:
            names = tuple(str(item) for item in self.channel_names)
            if len(names) != d or len(set(names)) != d:
                raise ValueError("channel_names must uniquely name every channel.")
            self.channel_names = names
        if int(self.original_ndim) not in (2, 3):
            raise ValueError("original_ndim must be 2 or 3.")
        self.values = values
        self.observation_mask = mask
        self.timestamps = timestamps

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.values.shape)  # type: ignore[return-value]

    def __len__(self) -> int:
        """Return the number of episodes for shared dataset validation."""
        return int(self.values.shape[0])

    @property
    def time_mask(self) -> np.ndarray:
        return np.any(self.observation_mask, axis=2)

    @property
    def lengths(self) -> np.ndarray:
        positions = np.arange(1, self.values.shape[1] + 1, dtype=int)
        return np.max(np.where(self.time_mask, positions[None, :], 0), axis=1)

    def take(self, indices: Sequence[int]) -> "MaskedSeriesBatch":
        idx = np.asarray(indices, dtype=int)
        return MaskedSeriesBatch(
            values=self.values[idx],
            observation_mask=self.observation_mask[idx],
            timestamps=self.timestamps[idx],
            channel_names=self.channel_names,
            original_ndim=self.original_ndim,
        )

    def align(self, max_length: int, n_channels: int) -> "MaskedSeriesBatch":
        """Align a prediction batch to the fitted horizon and channel schema."""
        n, T, d = self.values.shape
        if d > int(n_channels):
            raise ValueError(
                f"Prediction data has {d} channels; fitted schema has {n_channels}."
            )
        if T > int(max_length) and np.any(self.observation_mask[:, int(max_length) :, :]):
            raise ValueError(
                "Prediction data contains observations beyond the fitted horizon."
            )
        target_T, target_d = int(max_length), int(n_channels)
        values = np.zeros((n, target_T, target_d), dtype=float)
        mask = np.zeros((n, target_T, target_d), dtype=bool)
        timestamps = np.tile(np.arange(target_T, dtype=float), (n, 1))
        copy_T = min(T, target_T)
        values[:, :copy_T, :d] = self.values[:, :copy_T, :]
        mask[:, :copy_T, :d] = self.observation_mask[:, :copy_T, :]
        timestamps[:, :copy_T] = self.timestamps[:, :copy_T]
        return MaskedSeriesBatch(
            values=values,
            observation_mask=mask,
            timestamps=timestamps,
            channel_names=self.channel_names if d == target_d else None,
            original_ndim=self.original_ndim,
        )

    def legacy_array(self) -> np.ndarray:
        """Return a NaN-padded ndarray for legacy custom transformers."""
        values = np.where(self.observation_mask, self.values, np.nan)
        if self.original_ndim == 2 and values.shape[2] == 1:
            return values[:, :, 0]
        return values


ArrayLike = Union[np.ndarray, Sequence[np.ndarray], MaskedSeriesBatch]
DelayCostFunction = Callable[[float], float]


# ============================================================================
# Input and prefix utilities
# ============================================================================


def _canonicalize_X(X: ArrayLike) -> MaskedSeriesBatch:
    """Convert regular, missing, or ragged inputs into a masked batch."""
    if isinstance(X, MaskedSeriesBatch):
        return X
    if isinstance(X, np.ndarray) and X.dtype != object:
        array = np.asarray(X, dtype=float)
        if array.ndim not in (2, 3):
            raise ValueError("X must have shape (n,T), (n,T,d), or be ragged.")
        if any(size < 1 for size in array.shape):
            raise ValueError("X needs at least one episode, time point, and channel.")
        if np.isinf(array).any():
            raise ValueError("X contains infinity; use NaN only for declared missing observations.")
        original_ndim = int(array.ndim)
        if array.ndim == 2:
            array = array[:, :, None]
        if np.any(~np.any(np.isfinite(array[:, -1, :]), axis=1)):
            raise ValueError(
                "The final declared time point is entirely missing in at least one episode. "
                "Trailing missing observations must not silently shorten the observation budget; "
                "apply a declared causal missing-data policy before fitting or predicting."
            )
        return MaskedSeriesBatch(
            values=np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0),
            observation_mask=np.isfinite(array),
            original_ndim=original_ndim,
        )

    raw_values = X.tolist() if isinstance(X, np.ndarray) else list(X)
    if not raw_values:
        raise ValueError("X is empty.")
    arrays: List[np.ndarray] = []
    max_length = 0
    max_channels = 0
    all_univariate = True
    for item in raw_values:
        arr = np.asarray(item, dtype=float)
        if np.isinf(arr).any():
            raise ValueError("X contains infinity; use NaN only for declared missing observations.")
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError("Each ragged series must have shape (T,) or (T,d).")
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("Ragged series may not be empty.")
        if not np.isfinite(arr[-1]).any():
            raise ValueError(
                "The final declared time point of a ragged episode is entirely missing. "
                "A missing observation must not redefine its declared endpoint."
            )
        all_univariate = all_univariate and arr.shape[1] == 1
        max_length = max(max_length, int(arr.shape[0]))
        max_channels = max(max_channels, int(arr.shape[1]))
        arrays.append(arr)

    values = np.zeros((len(arrays), max_length, max_channels), dtype=float)
    mask = np.zeros_like(values, dtype=bool)
    for i, arr in enumerate(arrays):
        T_i, d_i = arr.shape
        finite = np.isfinite(arr)
        values[i, :T_i, :d_i] = np.where(finite, arr, 0.0)
        mask[i, :T_i, :d_i] = finite
    return MaskedSeriesBatch(
        values=values,
        observation_mask=mask,
        original_ndim=2 if all_univariate else 3,
    )


def _input_shape(X: ArrayLike) -> Tuple[int, int, int]:
    return _canonicalize_X(X).shape


def _take_samples(X: ArrayLike, indices: Sequence[int]) -> MaskedSeriesBatch:
    return _canonicalize_X(X).take(indices)


def _finite_scalar(value: Any, name: str) -> float:
    """Validate numerical controls without silently coercing strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite numerical scalar.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _integer_control(value: Any, name: str, minimum: int = 0) -> int:
    number = _finite_scalar(value, name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return int(number)


def default_prefix_transformer(X: ArrayLike, tau: int) -> np.ndarray:
    """Extract finite, mask-aware summary features from ``x[1:tau]``.

    Thirteen statistics are emitted per declared channel: mean, standard
    deviation, minimum, maximum, first and third quartiles, first and last
    observed values, change, timestamp-aware slope, observed fraction,
    relative last-observation position, and a channel-present indicator.
    """
    batch = _canonicalize_X(X)
    n, T, d = batch.shape
    t = _integer_control(tau, "tau", 1)
    if t < 1 or t > T:
        raise ValueError(f"tau must be an integer in [1,{T}], received {tau!r}.")
    rows: List[np.ndarray] = []
    for i in range(n):
        channel_rows: List[np.ndarray] = []
        for j in range(d):
            observed = np.flatnonzero(batch.observation_mask[i, :t, j])
            if len(observed) == 0:
                channel_rows.append(np.zeros(13, dtype=float))
                continue
            values = batch.values[i, observed, j]
            times = batch.timestamps[i, observed]
            centered = times - float(np.mean(times))
            denominator = float(np.dot(centered, centered))
            slope = (
                0.0
                if denominator <= 0.0
                else float(np.dot(centered, values - float(np.mean(values))) / denominator)
            )
            channel_rows.append(
                np.asarray(
                    [
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        np.quantile(values, 0.25),
                        np.quantile(values, 0.75),
                        values[0],
                        values[-1],
                        values[-1] - values[0],
                        slope,
                        float(len(observed)) / float(t),
                        float(observed[-1] + 1) / float(t),
                        1.0,
                    ],
                    dtype=float,
                )
            )
        rows.append(np.concatenate(channel_rows))
    features = np.vstack(rows)
    if not np.all(np.isfinite(features)):
        raise ValueError("Mask-aware prefix features contain NaN or infinity.")
    return features


def _validated_probability_matrix(probabilities: Any, *, min_classes: int = 2) -> np.ndarray:
    """Accept finite simplex rows; correct only floating-point scale deviations.

    Absolute tolerances are 1e-10 per component and 1e-8 for each row sum.
    Scores, counts and arbitrary positive weights are not probabilities.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < min_classes:
        raise ValueError("probabilities must be a non-empty (n_samples,n_classes) matrix.")
    if not np.isfinite(p).all() or np.any(p < -1e-10) or np.any(p > 1.0 + 1e-10):
        raise ValueError("Probability values must be finite and lie in [0,1].")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Each probability row must sum to one.")
    p = np.clip(p, 0.0, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _probability_vector(p: np.ndarray) -> np.ndarray:
    """Validate a one-dimensional probability simplex without epsilon clipping."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("A one-dimensional probability vector is required.")
    return _validated_probability_matrix(p[None, :])[0]


def _encoded_metric_inputs(y_encoded: Any, probabilities: Any) -> Tuple[np.ndarray, np.ndarray]:
    p = _validated_probability_matrix(probabilities)
    y = np.asarray(y_encoded)
    if y.ndim != 1 or len(y) != len(p) or y.dtype.kind not in "iuf":
        raise ValueError("Encoded labels must be a numeric vector aligned with probabilities.")
    if not np.isfinite(y).all() or np.any(y != np.floor(y)) or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("Encoded labels must be integers in [0,n_classes).")
    return y.astype(int), p


def _clip_probabilities_for_logloss(
    probabilities: np.ndarray, eps: float = 1e-15
) -> np.ndarray:
    """Clip only for numerical log-loss evaluation, never for decisions."""
    values = _validated_probability_matrix(probabilities)
    if not np.isfinite(eps) or not 0.0 < float(eps) < 0.5:
        raise ValueError("Log-loss epsilon must lie in (0,0.5).")
    clipped = np.clip(values, float(eps), 1.0 - float(eps))
    return clipped / clipped.sum(axis=1, keepdims=True)


# ============================================================================
# Reliability representation
# ============================================================================


@dataclass(frozen=True)
class ReliabilityWeights:
    w_pmax: float = 0.50
    w_margin: float = 0.30
    w_entropy: float = 0.10
    w_stability: float = 0.10


def _validate_reliability_weights(weights: ReliabilityWeights) -> None:
    values = np.asarray([_finite_scalar(value, name) for name, value in asdict(weights).items()])
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Reliability weights must be finite and non-negative.")
    if not np.any(values > 0.0):
        raise ValueError("At least one reliability weight must be positive.")


def reliability_features(
    p: np.ndarray, action_stability: float = 0.0
) -> Dict[str, float]:
    """Return the four GRACE reliability components for one prefix.

    ``action_stability`` is the Bayes-action stability indicator over the
    configured prefix window.  No label-stability or probability-drift proxy
    is used in GRACE.
    """
    p = _probability_vector(p)
    action_stability = _finite_scalar(action_stability, "action_stability")
    if not 0.0 <= action_stability <= 1.0:
        raise ValueError("action_stability must lie in [0,1].")
    ordered = np.sort(p)[::-1]
    positive = p[p > 0.0]
    entropy_normalized = float(
        -np.sum(positive * np.log(positive)) / np.log(len(p))
    )
    return {
        "pmax": float(ordered[0]),
        "margin": float(ordered[0] - ordered[1]),
        "entropy": entropy_normalized,
        "entropy_normalized": entropy_normalized,
        "action_stability": float(np.clip(action_stability, 0.0, 1.0)),
    }


def reliability_score(
    features: Mapping[str, float],
    n_classes: int,
    weights: ReliabilityWeights,
) -> float:
    """Return the bounded GRACE reliability score from theoretical components."""
    if n_classes < 2:
        raise ValueError("n_classes must be at least two.")
    _validate_reliability_weights(weights)
    entropy = float(
        features.get("entropy_normalized", features.get("entropy", 0.0))
    )
    terms = (
        (weights.w_pmax, float(features["pmax"])),
        (weights.w_margin, float(features["margin"])),
        (weights.w_entropy, 1.0 - entropy),
        (weights.w_stability, float(features["action_stability"])),
    )
    values = np.asarray([value for _, value in terms], dtype=float)
    if not np.isfinite(values).all() or np.any((values < -1e-10) | (values > 1.0 + 1e-10)):
        raise ValueError("Reliability components must be finite values in [0,1].")
    normalized_weights = np.asarray([weight for weight, _ in terms], dtype=float)
    normalized_weights /= float(np.max(normalized_weights))
    normalized_weights /= float(normalized_weights.sum())
    return float(np.dot(normalized_weights, np.clip(values, 0.0, 1.0)))


# ============================================================================
# Cost-aware decision model
# ============================================================================


@dataclass
class CostModel:
    alpha_delay: float
    tau_max: int
    misclassification_cost: np.ndarray
    delay_cost_fn: Optional[DelayCostFunction] = None
    observation_cost_fn: Optional[DelayCostFunction] = None
    alpha_observation: float = 0.0
    cost_scenarios: Optional[np.ndarray] = None
    scenario_weights: Optional[np.ndarray] = None
    delay_multipliers: Optional[np.ndarray] = None
    robust_mode: str = "nominal"
    cvar_alpha: float = 0.90

    def __post_init__(self) -> None:
        self.alpha_delay = float(self.alpha_delay)
        self.alpha_observation = float(self.alpha_observation)
        if isinstance(self.tau_max, bool) or not np.isfinite(self.tau_max) or int(self.tau_max) != self.tau_max:
            raise ValueError("tau_max must be a positive integer.")
        self.tau_max = int(self.tau_max)
        nominal = np.asarray(self.misclassification_cost, dtype=float)
        if not np.isfinite([self.alpha_delay, self.alpha_observation]).all() or self.alpha_delay < 0.0 or self.alpha_observation < 0.0 or self.tau_max < 1:
            raise ValueError(
                "alpha_delay and alpha_observation must be non-negative and tau_max positive."
            )
        if nominal.ndim != 2 or nominal.shape[0] < 2 or nominal.shape[0] != nominal.shape[1]:
            raise ValueError("misclassification_cost must be a square matrix.")
        if not np.all(np.isfinite(nominal)) or np.any(nominal < 0.0):
            raise ValueError("Misclassification costs must be finite and non-negative.")
        self.misclassification_cost = nominal
        if self.robust_mode not in {"nominal", "expected", "worst_case", "cvar"}:
            raise ValueError(
                "robust_mode must be 'nominal', 'expected', 'worst_case', or 'cvar'."
            )
        if not 0.0 < float(self.cvar_alpha) < 1.0:
            raise ValueError("cvar_alpha must lie strictly between zero and one.")

        if self.cost_scenarios is None:
            scenarios = nominal[None, :, :]
        else:
            scenarios = np.asarray(self.cost_scenarios, dtype=float)
            if scenarios.ndim == 2:
                scenarios = scenarios[None, :, :]
            if scenarios.ndim != 3 or scenarios.shape[1:] != nominal.shape:
                raise ValueError(
                    "cost_scenarios must have shape (n_scenarios,K,K)."
                )
            if not np.all(np.isfinite(scenarios)) or np.any(scenarios < 0.0):
                raise ValueError("Cost scenarios must be finite and non-negative.")
        n_scenarios = int(scenarios.shape[0])
        if n_scenarios < 1: raise ValueError("At least one cost scenario is required.")
        if self.scenario_weights is None:
            weights = np.full(n_scenarios, 1.0 / n_scenarios, dtype=float)
        else:
            weights = np.asarray(self.scenario_weights, dtype=float).reshape(-1)
            if len(weights) != n_scenarios:
                raise ValueError("scenario_weights must match cost_scenarios.")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("scenario_weights must be finite and non-negative.")
            maximum = float(np.max(weights))
            if maximum <= 0.0:
                raise ValueError("At least one scenario weight must be positive.")
            weights = weights / maximum
            weights = weights / float(weights.sum())
        if self.delay_multipliers is None:
            delay_multipliers = np.ones(n_scenarios, dtype=float)
        else:
            delay_multipliers = np.asarray(self.delay_multipliers, dtype=float).reshape(-1)
            if len(delay_multipliers) != n_scenarios:
                raise ValueError("delay_multipliers must match cost_scenarios.")
            if not np.all(np.isfinite(delay_multipliers)) or np.any(delay_multipliers < 0.0):
                raise ValueError("delay_multipliers must be finite and non-negative.")
        self.cost_scenarios = scenarios
        self.scenario_weights = weights
        self.delay_multipliers = delay_multipliers

    @property
    def n_scenarios(self) -> int:
        return int(np.asarray(self.cost_scenarios).shape[0])

    def delay_component(
        self, tau: Optional[int] = None, *, tau_fraction: Optional[float] = None
    ) -> float:
        if tau_fraction is None:
            if tau is None:
                raise ValueError("Supply tau or tau_fraction.")
            fraction = float(tau) / float(self.tau_max)
        else:
            fraction = float(tau_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0 + 1e-12:
            raise ValueError("tau_fraction must lie in [0,1].")
        fraction = float(np.clip(fraction, 0.0, 1.0))
        value = fraction if self.delay_cost_fn is None else float(self.delay_cost_fn(fraction))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("The delay-cost function must return a finite non-negative value.")
        return value

    def _aggregate(self, scenario_values: np.ndarray) -> np.ndarray:
        """Aggregate the first axis while preserving all trailing dimensions."""
        values = np.asarray(scenario_values, dtype=float)
        if values.ndim < 1 or values.shape[0] != self.n_scenarios:
            raise ValueError("Scenario axis does not match the configured cost model.")
        if not np.isfinite(values).all():
            raise ValueError("Aggregated scenario costs must be finite.")
        if self.robust_mode == "nominal":
            return values[0]
        if self.robust_mode == "expected":
            return np.tensordot(self.scenario_weights, values, axes=(0, 0))
        if self.robust_mode == "worst_case":
            return np.max(values, axis=0)

        # Discrete weighted upper-tail CVaR.  Boundary mass is split exactly.
        flat = values.reshape(self.n_scenarios, -1)
        result = np.zeros(flat.shape[1], dtype=float)
        tail_mass = 1.0 - float(self.cvar_alpha)
        for column in range(flat.shape[1]):
            order = np.argsort(flat[:, column])[::-1]
            remaining = tail_mass
            total = 0.0
            for scenario in order:
                take = min(remaining, float(self.scenario_weights[scenario]))
                total += take * float(flat[scenario, column])
                remaining -= take
                if remaining <= 0.0:
                    break
            result[column] = total / tail_mass
        return result.reshape(values.shape[1:])

    def delay_penalty(
        self, tau: Optional[int] = None, *, tau_fraction: Optional[float] = None
    ) -> float:
        component = self.delay_component(tau, tau_fraction=tau_fraction)
        if self.robust_mode == "nominal":
            return float(self.alpha_delay * component)
        penalties = self.alpha_delay * component * np.asarray(self.delay_multipliers)
        return float(self._aggregate(penalties))

    def observation_component(
        self, tau: Optional[int] = None, *, tau_fraction: Optional[float] = None
    ) -> float:
        """Return C_obs on its declared scale; zero when not modeled separately."""
        if tau_fraction is None:
            if tau is None:
                raise ValueError("Supply tau or tau_fraction.")
            fraction = float(tau) / float(self.tau_max)
        else:
            fraction = float(tau_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0 + 1e-12:
            raise ValueError("tau_fraction must lie in [0,1].")
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if self.observation_cost_fn is None:
            return 0.0
        value = float(self.observation_cost_fn(fraction))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "The observation-cost function must return a finite non-negative value."
            )
        return value

    def observation_penalty(
        self, tau: Optional[int] = None, *, tau_fraction: Optional[float] = None
    ) -> float:
        return float(
            self.alpha_observation
            * self.observation_component(tau, tau_fraction=tau_fraction)
        )

    def validate_time_cost_grid(self, fractions: Sequence[float]) -> Dict[str, Any]:
        """Check the declared time costs at the fractions used by a policy.

        Both components must be finite, non-negative and nondecreasing on this
        finite grid, as required by the reference cost specification. This
        check does not establish monotonicity between the supplied fractions.
        Callables must be deterministic functions of their fraction argument.
        """
        grid = np.asarray(fractions, dtype=float)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
            raise ValueError("Time-cost validation needs a non-empty finite fraction grid.")
        if np.any((grid < 0.0) | (grid > 1.0)):
            raise ValueError("Time-cost validation fractions must lie in [0,1].")
        grid = np.unique(grid)
        components: Dict[str, np.ndarray] = {}
        for name, function in (("delay", self.delay_component),
                               ("observation", self.observation_component)):
            values = np.asarray([function(tau_fraction=float(u)) for u in grid])
            # Allow only roundoff-scale reversals, without an absolute floor
            # that could hide a decreasing cost expressed in small units.
            local_scale = np.maximum(values[:-1], values[1:])
            tolerance = 32.0 * np.finfo(float).eps * local_scale
            if np.any(values[1:] < values[:-1] - tolerance):
                raise ValueError(f"The {name}-cost function must be nondecreasing on the decision grid.")
            components[name] = values
        with np.errstate(over="ignore", invalid="ignore"):
            multipliers = (
                np.ones(1) if self.robust_mode == "nominal"
                else np.asarray(self.delay_multipliers)
            )
            penalties = (
                self.alpha_delay * components["delay"][:, None]
                * multipliers[None, :]
                + self.alpha_observation * components["observation"][:, None]
            )
        if not np.isfinite(penalties).all():
            raise ValueError("Weighted time costs overflow at a declared decision fraction.")
        return {
            "scope": "declared_decision_fractions_only",
            "fractions": grid.tolist(),
            "delay_components": components["delay"].tolist(),
            "observation_components": components["observation"].tolist(),
            "nondecreasing": True,
        }

    def scenario_expected_action_risks(self, p: np.ndarray) -> np.ndarray:
        """Return action risks under the theoretical matrix convention L(a,k).

        The first matrix axis is the selected action and the second axis is the
        true class. Delay and observation costs are intentionally excluded.
        """
        p = _probability_vector(p)
        return np.einsum("k,sak->sa", p, self.cost_scenarios)

    def expected_action_risks(self, p: np.ndarray) -> np.ndarray:
        p = _probability_vector(p)
        if self.robust_mode == "nominal":
            return np.asarray(np.einsum("k,ak->a", p, self.misclassification_cost), dtype=float)
        return np.asarray(self._aggregate(self.scenario_expected_action_risks(p)), dtype=float)

    def scenario_expected_action_costs(
        self, p: np.ndarray, *, tau_fraction: float = 0.0
    ) -> np.ndarray:
        risks = self.scenario_expected_action_risks(p)
        delay = (
            self.alpha_delay
            * self.delay_component(tau_fraction=tau_fraction)
            * np.asarray(self.delay_multipliers)
        )
        observation = self.observation_penalty(tau_fraction=tau_fraction)
        return risks + delay[:, None] + observation

    def expected_action_costs(
        self, p: np.ndarray, *, tau_fraction: float = 0.0
    ) -> np.ndarray:
        if self.robust_mode == "nominal":
            return np.asarray(
                self.expected_action_risks(p)
                + self.delay_penalty(tau_fraction=tau_fraction)
                + self.observation_penalty(tau_fraction=tau_fraction),
                dtype=float,
            )
        return np.asarray(
            self._aggregate(
                self.scenario_expected_action_costs(p, tau_fraction=tau_fraction)
            ),
            dtype=float,
        )

    def bayes_action(self, p: np.ndarray, *, tau_fraction: float = 0.0) -> int:
        return int(np.argmin(self._action_comparison_costs(p, tau_fraction=tau_fraction)))

    def _action_comparison_costs(self, p: np.ndarray, *, tau_fraction: float = 0.0) -> np.ndarray:
        """Omit common offsets before comparing actions, avoiding false ties.

        Expected scenario aggregation is linear, so every time penalty is
        action-independent. Worst-case and CVaR aggregation are translation
        equivariant: observation cost and a shared scenario delay can also be
        omitted. Heterogeneous scenario delays remain inside those nonlinear
        aggregations, since removing them would change the decision problem.
        """
        delay = self.delay_component(tau_fraction=tau_fraction)
        self.observation_component(tau_fraction=tau_fraction)
        multipliers = np.asarray(self.delay_multipliers)
        if (self.robust_mode in {"nominal", "expected"}
                or np.all(multipliers == multipliers[0])):
            return self.expected_action_risks(p)
        return np.asarray(self._aggregate(
            self.scenario_expected_action_risks(p)
            + (self.alpha_delay * delay * multipliers)[:, None]
        ), dtype=float)

    def _class_index(self, value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be an integer class index.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer class index.") from exc
        if not np.isfinite(number) or int(number) != number or not 0 <= number < self.misclassification_cost.shape[0]:
            raise ValueError(f"{name} is outside the configured class index range.")
        return int(number)

    def actual_misclassification_cost(self, y_index: int, action_index: int) -> float:
        y_index = self._class_index(y_index, "y_index")
        action_index = self._class_index(action_index, "action_index")
        if self.robust_mode == "nominal":
            return float(self.misclassification_cost[int(action_index), int(y_index)])
        values = np.asarray(self.cost_scenarios)[:, int(action_index), int(y_index)]
        return float(self._aggregate(values))

    def total_cost(
        self,
        y_index: int,
        action_index: int,
        tau: Optional[int] = None,
        *,
        tau_fraction: Optional[float] = None,
    ) -> float:
        y_index = self._class_index(y_index, "y_index")
        action_index = self._class_index(action_index, "action_index")
        component = self.delay_component(tau, tau_fraction=tau_fraction)
        observation = self.observation_penalty(tau_fraction=(
            float(tau) / float(self.tau_max) if tau_fraction is None else float(tau_fraction)
        ))
        if self.robust_mode == "nominal":
            return float(
                self.misclassification_cost[int(action_index), int(y_index)]
                + self.alpha_delay * component
                + observation
            )
        scenario_values = (
            np.asarray(self.cost_scenarios)[:, int(action_index), int(y_index)]
            + self.alpha_delay * component * np.asarray(self.delay_multipliers)
            + observation
        )
        return float(self._aggregate(scenario_values))

    def expected_total_cost(
        self,
        p: np.ndarray,
        tau: Optional[int] = None,
        *,
        tau_fraction: Optional[float] = None,
    ) -> float:
        if tau_fraction is None:
            if tau is None:
                raise ValueError("Supply tau or tau_fraction.")
            tau_fraction = float(tau) / float(self.tau_max)
        return float(np.min(self.expected_action_costs(p, tau_fraction=float(tau_fraction))))


# ============================================================================
# GRACE estimator
# ============================================================================

class _GRACECoreBase(ClassifierMixin, BaseEstimator):
    """Shared implementation utilities for the GRACE estimator."""

    def _resolve_weights(self) -> ReliabilityWeights:
        if self.reliability_weights is None:
            weights = ReliabilityWeights()
        elif isinstance(self.reliability_weights, ReliabilityWeights):
            weights = self.reliability_weights
        elif isinstance(self.reliability_weights, Mapping):
            weights = ReliabilityWeights(**dict(self.reliability_weights))
        else:
            raise TypeError("reliability_weights must be a mapping or ReliabilityWeights.")
        _validate_reliability_weights(weights)
        return weights

    def _resolve_tau_grid(self, T: int) -> np.ndarray:
        raw = (0.20, 0.35, 0.50, 0.65, 0.80, 1.00) if self.decision_points is None else self.decision_points
        values = np.asarray([_finite_scalar(value, "decision_points") for value in raw], dtype=float)
        if values.ndim != 1 or len(values) < 1 or np.any(~np.isfinite(values)):
            raise ValueError("decision_points must be a non-empty finite sequence.")
        if np.any(values <= 0.0):
            raise ValueError("Decision times must be positive.")
        if np.any(values < 1.0) and np.any(values > 1.0):
            raise ValueError("Do not mix fractional and absolute decision times.")
        fractional = bool(np.all(values <= 1.0))
        self.tau_input_mode_ = "fractional" if fractional else "absolute"
        if fractional:
            resolved = np.ceil(values * int(T)).astype(int)
        else:
            if not np.array_equal(values, np.round(values)):
                raise ValueError("Absolute decision times must be integer-valued.")
            resolved = np.round(values).astype(int)
        if np.any(resolved < 1) or np.any(resolved > int(T)):
            raise ValueError(f"Resolved decision times must lie in [1,{T}].")
        if np.any(np.diff(resolved) <= 0):
            raise ValueError("Decision times must be strictly increasing after resolution.")
        # Theory-conformant invariant: the final decision point is always the full horizon.
        # ``force_terminal_full`` is retained only for backward configuration compatibility.
        # Setting it to False no longer disables the terminal rule.
        if resolved[-1] != int(T):
            resolved = np.append(resolved, int(T))
        self.terminal_rule_forced_ = True
        return resolved.astype(int)

    def _class_reorder(self) -> np.ndarray:
        """Map supplied loss-matrix axes to the fitted encoded class order.

        ``cost_class_order`` describes matrix axes; it does not override the
        LabelEncoder class order used for exact-risk tie breaking.
        """
        K = self.n_classes_
        if self.cost_class_order is None:
            return np.arange(K, dtype=int)
        supplied = list(self.cost_class_order)
        if len(supplied) != K or len({str(value) for value in supplied}) != K:
            raise ValueError("cost_class_order must uniquely list all classes.")
        model_keys = [str(value) for value in self.classes_]
        supplied_keys = [str(value) for value in supplied]
        if set(supplied_keys) != set(model_keys):
            raise ValueError("cost_class_order must match the observed classes.")
        return np.asarray([supplied_keys.index(key) for key in model_keys], dtype=int)

    def _validate_cost_configuration(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        K = self.n_classes_
        nominal = (
            np.ones((K, K), dtype=float) - np.eye(K, dtype=float)
            if self.misclassification_cost is None
            else np.asarray(self.misclassification_cost, dtype=float)
        )
        if nominal.shape != (K, K) or np.any(~np.isfinite(nominal)) or np.any(nominal < 0):
            raise ValueError(f"misclassification_cost must be finite, non-negative, shape ({K},{K}).")
        order = self._class_reorder()
        nominal = nominal[np.ix_(order, order)]
        if self.cost_scenarios is None:
            return nominal, None
        scenarios = np.asarray(self.cost_scenarios, dtype=float)
        if scenarios.ndim == 2:
            scenarios = scenarios[None, :, :]
        if scenarios.ndim != 3 or scenarios.shape[1:] != (K, K):
            raise ValueError(f"cost_scenarios must have shape (S,{K},{K}).")
        if np.any(~np.isfinite(scenarios)) or np.any(scenarios < 0.0):
            raise ValueError("cost_scenarios must be finite and non-negative.")
        scenarios = scenarios[:, order, :][:, :, order]
        return nominal, scenarios

    def _split_has_all_classes(self, y: np.ndarray) -> bool:
        return len(np.unique(y)) == self.n_classes_

    def _safe_stratified_split(
        self,
        indices: np.ndarray,
        y: np.ndarray,
        *,
        train_fraction: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=int)
        if len(indices) < 2:
            raise ValueError("At least two rows are required for a split.")
        stratify: Optional[np.ndarray] = y[indices]
        counts = np.bincount(y[indices], minlength=self.n_classes_)
        if np.any(counts < 2):
            stratify = None
        first, second = train_test_split(
            indices,
            train_size=float(train_fraction),
            random_state=int(seed),
            stratify=stratify,
        )
        return np.sort(first), np.sort(second)

    def _temporal_gap(self) -> int:
        """Total ordered-window gap used by temporal validation splits."""
        return int(self.crossfit_purge) + int(self.crossfit_embargo)

    def _purge_temporal_pair(self, train, valid, role):
        train, valid = np.asarray(train,int), np.asarray(valid,int)
        if not len(train) or not len(valid): raise ValueError(f"Empty {role} partition.")
        meta = getattr(self,"temporal_metadata_",None)
        if meta is None: return train,valid
        active = self._active_temporal_rows_
        boundary = np.min(meta['information_start'][active[valid]])
        keep = (meta['information_end'][active[train]] < boundary) & (meta['episode_time'][active[train]] < np.min(meta['episode_time'][active[valid]]))
        removed, train = train[~keep], train[keep]
        if not len(train): raise ValueError(f"Information-interval purging leaves no {role} training rows.")
        self.temporal_split_audit_.append({'role':role,'train':active[train].tolist(),'validation':active[valid].tolist(),
            'interval_purged_rows':active[removed].tolist(),'latest_training_information_end_ns':int(np.max(meta['information_end'][active[train]])),
            'earliest_validation_information_start_ns':int(boundary)})
        return train,valid

    def _isolate_temporal_roles(self, partitions):
        if self.split_strategy == 'chronological':
            keys = list(partitions)
            for j,key in enumerate(keys[:-1]):
                later = np.concatenate([partitions[k] for k in keys[j+1:]])
                partitions[key], _ = self._purge_temporal_pair(partitions[key],later,'role_'+key)
        return partitions

    def _crossfit_partitions(
        self, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Reserve gated-acceptance development and validation data.

        For chronological data, windows between adjacent roles are discarded
        according to ``crossfit_purge + crossfit_embargo``.
        """
        indices = np.arange(len(y), dtype=int)
        if len(indices) < 8:
            raise ValueError("Too few rows for cross-fitting and gated-acceptance validation.")
        if self.split_strategy == "group":
            if groups is None:
                raise ValueError("groups are required when split_strategy='group'.")
            first_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=float(self.gated_acceptance_size),
                random_state=(int(self.random_state) + 101) % (2 ** 32),
            )
            development, acceptance = next(
                first_splitter.split(indices, y, groups=groups)
            )
            validation_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=float(self.gated_acceptance_validation_fraction),
                random_state=(int(self.random_state) + 102) % (2 ** 32),
            )
            dev_local, val_local = next(
                validation_splitter.split(acceptance, y[acceptance], groups=groups[acceptance])
            )
            acceptance_development = acceptance[dev_local]
            acceptance_validation = acceptance[val_local]
        elif self.split_strategy == "chronological":
            gap = self._temporal_gap()
            acceptance_count = max(2, int(round(len(indices) * float(self.gated_acceptance_size))))
            acceptance_count = min(acceptance_count, len(indices) - 2)
            acceptance_start = len(indices) - acceptance_count
            development_stop = acceptance_start - gap
            if development_stop < 2:
                raise ValueError(
                    "Temporal purge+embargo leaves too few predictor-development rows; "
                    "provide more historical data or prespecify different role shares."
                )
            development = indices[:development_stop]
            acceptance = indices[acceptance_start:]
            validation_count = max(1, int(round(len(acceptance) * float(self.gated_acceptance_validation_fraction))))
            validation_count = min(validation_count, len(acceptance) - 1)
            validation_start = len(acceptance) - validation_count
            acceptance_dev_stop = validation_start - gap
            if acceptance_dev_stop < 1:
                raise ValueError(
                    "Temporal purge+embargo leaves no gated-acceptance development rows; "
                    "provide more historical data or prespecify different role shares."
                )
            acceptance_development = acceptance[:acceptance_dev_stop]
            acceptance_validation = acceptance[validation_start:]
        else:
            development, acceptance = self._safe_stratified_split(
                indices,
                y,
                train_fraction=1.0 - float(self.gated_acceptance_size),
                seed=(int(self.random_state) + 101) % (2 ** 32),
            )
            acceptance_development, acceptance_validation = self._safe_stratified_split(
                acceptance,
                y,
                train_fraction=1.0 - float(self.gated_acceptance_validation_fraction),
                seed=(int(self.random_state) + 102) % (2 ** 32),
            )
        return {
            "development": np.asarray(development, dtype=int),
            "gated_acceptance_development": np.asarray(acceptance_development, dtype=int),
            "gated_acceptance_validation": np.asarray(acceptance_validation, dtype=int),
        }

    def _holdout_partitions(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Create disjoint prefix, calibration, reliability, and acceptance subsets."""
        n = len(y)
        idx = np.arange(n, dtype=int)
        n_cal = max(1, int(round(n * float(self.calibration_size))))
        n_rel = max(1, int(round(n * float(self.reliability_size))))
        n_accept = max(2, int(round(n * float(self.gated_acceptance_size))))
        if self.split_strategy == "chronological":
            gap = self._temporal_gap()
            n_base = n - n_cal - n_rel - n_accept - 3 * gap
            if n_base < 1:
                raise ValueError(
                    "Holdout fractions plus temporal purge+embargo leave no prefix-prediction rows."
                )
            cursor = 0
            prefix = idx[cursor:cursor + n_base]; cursor += n_base + gap
            calibration = idx[cursor:cursor + n_cal]; cursor += n_cal + gap
            reliability = idx[cursor:cursor + n_rel]; cursor += n_rel + gap
            acceptance = idx[cursor:]
            val_n = max(1, int(round(len(acceptance) * float(self.gated_acceptance_validation_fraction))))
            val_n = min(val_n, len(acceptance) - 1)
            val_start = len(acceptance) - val_n
            dev_stop = val_start - gap
            if dev_stop < 1:
                raise ValueError(
                    "Temporal purge+embargo leaves no gated-acceptance development rows in holdout mode."
                )
            acceptance_development = acceptance[:dev_stop]
            acceptance_validation = acceptance[val_start:]
        else:
            n_base = n - n_cal - n_rel - n_accept
            if n_base < 1:
                raise ValueError("Holdout fractions leave no prefix-prediction rows.")
            prefix, rest = self._safe_stratified_split(
                idx, y, train_fraction=float(n_base) / n, seed=int(self.random_state)
            )
            cal_fraction = float(n_cal) / float(len(rest))
            calibration, policy = self._safe_stratified_split(
                rest, y, train_fraction=cal_fraction, seed=(int(self.random_state) + 1) % (2 ** 32)
            )
            rel_fraction = float(n_rel) / float(len(policy))
            reliability, acceptance = self._safe_stratified_split(
                policy, y, train_fraction=rel_fraction, seed=(int(self.random_state) + 2) % (2 ** 32)
            )
            acceptance_development, acceptance_validation = self._safe_stratified_split(
                acceptance,
                y,
                train_fraction=1.0 - float(self.gated_acceptance_validation_fraction),
                seed=(int(self.random_state) + 3) % (2 ** 32),
            )
        for role, rows in {"prefix_prediction": prefix, "calibration": calibration}.items():
            if not self._split_has_all_classes(y[rows]):
                raise ValueError(f"The {role} holdout subset is missing an outcome class.")
        return {
            "prefix_prediction": np.asarray(prefix, dtype=int),
            "calibration": np.asarray(calibration, dtype=int),
            "reliability": np.asarray(reliability, dtype=int),
            "gated_acceptance_development": np.asarray(acceptance_development, dtype=int),
            "gated_acceptance_validation": np.asarray(acceptance_validation, dtype=int),
        }

    def _transform(self, X: ArrayLike, tau: int) -> np.ndarray:
        batch = _canonicalize_X(X)
        transformer = self.prefix_transformer_
        if transformer is default_prefix_transformer:
            raw_input: Any = batch
        elif self.transformer_input == "masked":
            # A prefix view may legitimately have no observations yet for a row.
            # Copy the validated batch before slicing, without changing the input.
            raw_input = copy.copy(batch)
            raw_input.values = batch.values[:, :int(tau)].copy()
            raw_input.observation_mask = batch.observation_mask[:, :int(tau)].copy()
            raw_input.timestamps = batch.timestamps[:, :int(tau)].copy()
        else:
            # Callbacks receive no future values, even if they ignore tau.
            raw_input = batch.legacy_array()[:, :int(tau)].copy()
        features = np.asarray(transformer(raw_input, int(tau)), dtype=float)
        if features.ndim != 2 or features.shape[0] != batch.shape[0]:
            raise ValueError("prefix_transformer must return (n_samples,n_features).")
        if np.any(~np.isfinite(features)):
            raise ValueError(
                "prefix_transformer returned NaN or infinity; custom transformers "
                "must handle padded/missing observations explicitly."
            )
        return features

    def _fit_prefit_calibrator(self, estimator, X_cal, y_cal):
        if self.calibration_method == "none":
            estimator.effective_calibration_scheme_ = "disabled_ablation"
            return estimator
        if self.calibration_method == "temperature":
            return _LogisticTemperatureCalibrator(estimator).fit(np.asarray(X_cal, float), np.asarray(y_cal, int))
        if FrozenEstimator is None:
            raise RuntimeError("Calibration requires scikit-learn >=1.6.")
        if set(np.unique(y_cal)) != set(estimator.classes_):
            raise ValueError("Calibration rows must contain every fitted class.")
        rows = np.arange(len(y_cal))
        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=self.calibration_method,
            cv=[(rows, rows)], ensemble=False).fit(np.asarray(X_cal,float), np.asarray(y_cal,int))
        calibrated.effective_calibration_scheme_ = "explicit_frozen_holdout"
        calibrated.n_calibration_rows_ = len(rows)
        return calibrated

    def _fit_calibrator_rows(
        self,
        features: np.ndarray,
        y: np.ndarray,
        template: BaseEstimator,
        seed: int,
        groups: Optional[np.ndarray] = None,
    ) -> CalibratedClassifierCV:
        """Fit a calibrated classifier without crossing the declared design unit."""
        if not self._split_has_all_classes(y):
            raise ValueError("Calibrator training rows must contain every class.")
        if self.split_strategy == "chronological":
            n = len(y)
            gap = self._temporal_gap()
            preferred = int(round(0.80 * n))
            candidates = sorted(
                range(max(2, int(0.55 * n)), min(n - 1, int(0.90 * n)) + 1),
                key=lambda value: abs(value - preferred),
            )
            cut = next(
                (
                    value
                    for value in candidates
                    if value + gap < n
                    and self._split_has_all_classes(y[:value])
                    and self._split_has_all_classes(y[value + gap :])
                ),
                None,
            )
            if cut is None:
                raise ValueError(
                    "A purged chronological calibration split containing every class could not be formed."
                )
            model = clone(template).fit(features[:cut], y[:cut])
            return self._fit_prefit_calibrator(model, features[cut + gap :], y[cut + gap :])

        if self.split_strategy == "group":
            if groups is None:
                raise ValueError("groups are required for group-aware calibration.")
            group_array = np.asarray(groups)
            unique_groups = np.unique(group_array)
            class_group_counts = [
                len(np.unique(group_array[y == cls])) for cls in range(self.n_classes_)
            ]
            folds = min(
                int(self.crossfit_inner_folds),
                int(len(unique_groups)),
                int(min(class_group_counts)),
            )
            if folds < 2:
                raise ValueError(
                    "Group-aware calibration requires at least two distinct groups for every class."
                )
            splitter = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=int(seed)
            )
            splits = list(splitter.split(np.zeros(len(y)), y, group_array))
            for train, valid in splits:
                if set(group_array[train]).intersection(set(group_array[valid])):
                    raise RuntimeError("Internal group calibration split leaked a group.")
            calibrator = CalibratedClassifierCV(
                estimator=clone(template), method=self.calibration_method, cv=splits
            )
            calibrator.fit(features, y)
            return calibrator

        counts = np.bincount(y, minlength=self.n_classes_)
        folds = min(int(self.crossfit_inner_folds), int(np.min(counts)))
        if folds < 2:
            raise ValueError("At least two rows per class are needed for calibration.")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        calibrator = CalibratedClassifierCV(
            estimator=clone(template), method=self.calibration_method, cv=cv
        )
        calibrator.fit(features, y)
        return calibrator

    def _ordered_predict_proba(
        self, calibrator: CalibratedClassifierCV, features: np.ndarray
    ) -> np.ndarray:
        raw = _validated_probability_matrix(calibrator.predict_proba(features))
        columns = np.asarray(calibrator.classes_)
        if (columns.ndim != 1 or len(raw) != len(features) or raw.shape[1] != len(columns)
                or not np.array_equal(np.sort(columns), np.arange(self.n_classes_))):
            raise ValueError("Prefix calibrator class columns must be a permutation of fitted class indices.")
        ordered = np.zeros((len(features), self.n_classes_), dtype=float)
        ordered[:, columns.astype(int)] = raw
        return ordered

    def _outer_splits(
        self, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        self._outer_split_exclusions_ = []
        self._outer_split_fold_ids_ = []
        if self.split_strategy == "group":
            if groups is None:
                raise ValueError("groups are required when split_strategy='group'.")
            unique_groups = np.unique(groups)
            folds = min(int(self.crossfit_folds), len(unique_groups))
            if folds < 2:
                raise ValueError("Group cross-fitting requires at least two groups.")
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=int(self.random_state),
            )
            return [
                (train, valid)
                for train, valid in splitter.split(np.zeros(len(y)), y, groups)
            ]
        if self.split_strategy == "chronological":
            splitter = TimeSeriesSplit(
                n_splits=int(self.crossfit_folds), gap=self._temporal_gap()
            )
            splits = []
            for fold, (train, valid) in enumerate(splitter.split(np.arange(len(y)))):
                try:
                    pair = self._purge_temporal_pair(train, valid, "predictor_outer")
                except ValueError as exc:
                    # An early fold can have insufficient support-isolated
                    # history while later expanding windows remain usable.
                    self._outer_split_exclusions_.append({
                        "fold": int(fold), "status": "excluded_infeasible",
                        "partition_stage": "outer_interval_purge", "reason": str(exc),
                        "outer_training_rows_before_purge": int(len(train)),
                        "outer_training_rows": 0, "outer_validation_rows": int(len(valid)),
                        "interval_purged_rows": self._active_temporal_rows_[train].tolist(),
                    })
                    continue
                splits.append(pair)
                self._outer_split_fold_ids_.append(int(fold))
            return splits
        counts = np.bincount(y, minlength=self.n_classes_)
        folds = min(int(self.crossfit_folds), int(np.min(counts)))
        if folds < 2:
            raise ValueError("Cross-fitting needs at least two observations per class.")
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=int(self.random_state)
        )
        return [(train, valid) for train, valid in splitter.split(np.zeros(len(y)), y)]

    def _fit_crossfit_predictors(
        self,
        X: MaskedSeriesBatch,
        y: np.ndarray,
        template: BaseEstimator,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, M, K = len(y), len(self.tau_grid_), self.n_classes_
        oof = np.full((n, M, K), np.nan, dtype=float)
        splits = self._outer_splits(y, groups)
        self.crossfit_splits_ = [
            {"train": train.copy(), "validation": valid.copy()} for train, valid in splits
        ]
        self.base_models_ = []
        self.calibrators_ = []
        self.feature_dimensions_ = []
        for m, tau in enumerate(self.tau_grid_):
            features = self._transform(X, int(tau))
            final_base = clone(template).fit(features, y)
            final_calibrator = self._fit_calibrator_rows(
                features, y, template, (int(self.random_state) + 1000 + m) % (2 ** 32), groups
            )
            self.base_models_.append(final_base)
            self.calibrators_.append(final_calibrator)
            self.feature_dimensions_.append(int(features.shape[1]))
            for fold, (train, valid) in enumerate(splits):
                if not self._split_has_all_classes(y[train]):
                    continue
                try:
                    calibrator = self._fit_calibrator_rows(
                        features[train],
                        y[train],
                        template,
                        (int(self.random_state) + 10000 + 100 * m + fold) % (2 ** 32),
                        None if groups is None else groups[train],
                    )
                except ValueError:
                    continue
                oof[valid, m] = self._ordered_predict_proba(
                    calibrator, features[valid]
                )
        available = np.all(np.isfinite(oof), axis=(1, 2))
        if int(np.sum(available)) < max(6, self.n_classes_ * 2):
            raise ValueError(
                "Cross-fitting produced too few complete probability trajectories; "
                "increase data, reduce folds, or use fit_mode='holdout'."
            )
        return oof, available

    def _fit_holdout_predictors(
        self,
        X: MaskedSeriesBatch,
        y: np.ndarray,
        partitions: Mapping[str, np.ndarray],
        template: BaseEstimator,
    ) -> None:
        self.base_models_ = []
        self.calibrators_ = []
        self.feature_dimensions_ = []
        base_rows, cal_rows = partitions["prefix_prediction"], partitions["calibration"]
        for tau in self.tau_grid_:
            base_features = self._transform(X.take(base_rows), int(tau))
            cal_features = self._transform(X.take(cal_rows), int(tau))
            model = clone(template).fit(base_features, y[base_rows])
            calibrator = self._fit_prefit_calibrator(model, cal_features, y[cal_rows])
            self.base_models_.append(model)
            self.calibrators_.append(calibrator)
            self.feature_dimensions_.append(int(base_features.shape[1]))

    def _probability_grid(self, X: ArrayLike) -> np.ndarray:
        probabilities = []
        for tau, calibrator in zip(self.tau_grid_, self.calibrators_):
            features = self._transform(X, int(tau))
            probabilities.append(self._ordered_predict_proba(calibrator, features))
        return np.stack(probabilities, axis=1)

    def _decision_geometry(
        self, X: MaskedSeriesBatch
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lengths = X.lengths.astype(int)
        M = len(self.tau_grid_)
        terminal = np.searchsorted(self.tau_grid_, lengths, side="left")
        terminal = np.clip(terminal, 0, M - 1).astype(int)
        valid = np.arange(M)[None, :] <= terminal[:, None]
        effective_times = np.minimum(self.tau_grid_[None, :], lengths[:, None])
        fractions = effective_times / lengths[:, None].astype(float)
        return effective_times.astype(int), fractions.astype(float), terminal, valid

    def _bayes_actions(
        self, probability_grid: np.ndarray, tau_fractions: np.ndarray
    ) -> np.ndarray:
        n, M, _ = probability_grid.shape
        actions = np.zeros((n, M), dtype=int)
        for i in range(n):
            for m in range(M):
                actions[i, m] = self.cost_model_.bayes_action(
                    probability_grid[i, m], tau_fraction=float(tau_fractions[i, m])
                )
        return actions

    @staticmethod
    def _positive_probability(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
        if hasattr(estimator, "predict_proba"):
            raw = _validated_probability_matrix(estimator.predict_proba(X), min_classes=1)
            classes = np.asarray(estimator.classes_)
            if (classes.ndim != 1 or len(raw) != len(X) or raw.shape[1] != len(classes)
                    or len(np.unique(classes)) != len(classes) or not np.isin(classes, [0, 1]).all()):
                raise ValueError("Binary gate probabilities have invalid shape or class mapping.")
            match = np.flatnonzero(classes == 1)
            return raw[:, match[0]] if len(match) else np.zeros(len(X), dtype=float)
        if hasattr(estimator, "decision_function"):
            classes = np.asarray(getattr(estimator, "classes_", []))
            if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
                raise ValueError("Binary decision scores require fitted classes 0 and 1.")
            score = np.asarray(estimator.decision_function(X), dtype=float)
            if score.shape != (len(X),) or not np.isfinite(score).all():
                raise ValueError("Binary decision scores must be finite and aligned.")
            # Binary decision_function is positive for classes_[1].
            if classes[1] != 1:
                score = -score
            return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
        predicted = np.asarray(estimator.predict(X))
        if predicted.shape != (len(X),) or not np.isin(predicted, [0, 1]).all():
            raise ValueError("Binary gate predictions must be aligned class indices 0 or 1.")
        return (predicted == 1).astype(float)

    def _fit_binary(
        self,
        template: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> BaseEstimator:
        unique = np.unique(y)
        estimator: BaseEstimator
        if len(unique) == 1:
            estimator = DummyClassifier(strategy="constant", constant=int(unique[0]))
        else:
            estimator = clone(template)
        parameters = inspect.signature(estimator.fit).parameters
        if sample_weight is not None and "sample_weight" in parameters:
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            estimator.fit(X, y)
        return estimator

    def _sample_level_splits(
        self,
        y: np.ndarray,
        n_folds: int,
        seed_offset: int,
        groups: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(y)
        if self.split_strategy == "group":
            if groups is None:
                raise ValueError("Reliability groups are required for group cross-fitting.")
            folds = min(int(n_folds), len(np.unique(groups)))
            if folds < 2:
                raise ValueError("At least two groups are required.")
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=(int(self.random_state) + seed_offset) % (2 ** 32),
            )
            return [
                (a, b) for a, b in splitter.split(np.zeros(n), y, groups)
            ]
        if self.split_strategy == "chronological":
            folds = min(int(n_folds), max(2, n // 4))
            splitter = TimeSeriesSplit(n_splits=folds, gap=self._temporal_gap())
            surviving = []
            for fold, (train, held) in enumerate(splitter.split(np.arange(n))):
                try:
                    surviving.append(self._purge_temporal_pair(train, held, "policy_crossfit"))
                except ValueError as exc:
                    if not hasattr(self, "policy_split_diagnostics_"):
                        self.policy_split_diagnostics_ = []
                    self.policy_split_diagnostics_.append({
                        "fold": int(fold), "status": "excluded_infeasible",
                        "reason": str(exc), "training_rows": int(len(train)),
                        "held_out_rows": int(len(held)),
                    })
            return surviving
        counts = np.bincount(y, minlength=self.n_classes_)
        positive = counts[counts > 0]
        folds = min(int(n_folds), int(np.min(positive)))
        if folds < 2:
            splitter = KFold(n_splits=2, shuffle=True, random_state=(int(self.random_state) + seed_offset) % (2 ** 32))
            return [(a, b) for a, b in splitter.split(np.arange(n))]
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=(int(self.random_state) + seed_offset) % (2 ** 32),
        )
        return [(a, b) for a, b in splitter.split(np.zeros(n), y)]

    def _validate_prediction_input(self, X: ArrayLike) -> MaskedSeriesBatch:
        batch = _canonicalize_X(X)
        if batch.channel_names is not None:
            expected = tuple(self.channel_names_[: batch.shape[2]])
            if tuple(batch.channel_names) != expected:
                raise ValueError("Prediction channel_names do not match the fitted schema prefix.")
        return batch.align(self.T_, self.n_channels_in_)

    def predict_early(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        labels, times, _, traces = self._predict_internal(X)
        return labels, times, traces

    def predict_early_proba(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        return self._predict_internal(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._predict_internal(X)[0]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self._predict_internal(X)[2]

    def set_safe_mode(self, enabled: bool = True, reason: Optional[str] = None) -> "GRACE":
        check_is_fitted(self, attributes=["gated_acceptance_schedule_"])
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean.")
        if not enabled and bool(getattr(self, "gated_acceptance_refit_required_", False)):
            raise ValueError("Refit the gated-acceptance rule before disabling recalibration safe mode.")
        self.safe_mode_ = bool(enabled)
        self.safe_mode_reason_ = str(reason) if enabled and reason is not None else None
        return self

    @staticmethod
    def _population_stability_index(
        reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        reference = np.asarray(reference, dtype=float)
        current = np.asarray(current, dtype=float)
        if len(reference) < 2 or len(current) < 1:
            return 0.0
        edges = np.unique(
            np.quantile(reference, np.linspace(0.0, 1.0, int(n_bins) + 1))
        )
        if len(edges) < 3:
            scale = max(abs(float(reference[0])), 1.0) * 1e-6
            edges = np.asarray([-np.inf, reference[0] - scale, reference[0] + scale, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        ref_count, _ = np.histogram(reference, bins=edges)
        cur_count, _ = np.histogram(current, bins=edges)
        eps = 1e-6
        ref_prop = np.clip(ref_count / max(1, len(reference)), eps, None)
        cur_prop = np.clip(cur_count / max(1, len(current)), eps, None)
        return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))

@dataclass
class GatedAcceptanceSchedule:
    """Frozen GRACE Gated Acceptance configuration selected on development data."""

    tau_grid: np.ndarray
    family: str
    active_features: Tuple[str, ...]
    action_stability_window: int
    thresholds_by_feature: Dict[str, np.ndarray]
    learned_threshold_by_tau: np.ndarray
    enabled_by_tau: np.ndarray
    validation_stats: List[Dict[str, Any]]
    development_objective: float
    validation_objective: float
    learned_history_mode: str = "soft"

    def _index(self, tau: int) -> int:
        matches = np.flatnonzero(self.tau_grid == int(tau))
        if len(matches) != 1:
            raise ValueError(f"tau={tau} is not a trained decision point.")
        return int(matches[0])

    def is_enabled(self, tau: int) -> bool:
        return bool(self.enabled_by_tau[self._index(tau)])

    def threshold(self, tau: int) -> float:
        """Return the principal acceptance threshold for reporting."""
        idx = self._index(tau)
        if self.family == "learned":
            return float(self.learned_threshold_by_tau[idx])
        values = self.thresholds_by_feature.get("pmax")
        return float(values[idx]) if values is not None else float("-inf")

    def as_dict(self) -> Dict[str, Any]:
        def clean(values: np.ndarray) -> List[Optional[float]]:
            return [float(value) if np.isfinite(value) else None for value in values]

        return {
            "schema": "grace-gated-acceptance",
            "architecture": "GRACE-current-prefix",
            "component": "GRACE Gated Acceptance",
            "gated_acceptance_family": self.family,
            "active_features": list(self.active_features),
            "action_stability_window": int(self.action_stability_window),
            "learned_history_mode": self.learned_history_mode,
            "decision_points": self.tau_grid.astype(int).tolist(),
            "enabled_by_decision_point": self.enabled_by_tau.astype(bool).tolist(),
            "thresholds_by_feature": {
                key: clean(np.asarray(value, dtype=float))
                for key, value in self.thresholds_by_feature.items()
            },
            "threshold_semantics_by_feature": {
                key: ["positive_infinity" if np.isposinf(x) else "negative_infinity" if np.isneginf(x) else "finite"
                      for x in value] for key, value in self.thresholds_by_feature.items()
            },
            "learned_acceptance_threshold_by_decision_point": clean(
                self.learned_threshold_by_tau
            ),
            "learned_threshold_semantics": ["positive_infinity" if np.isposinf(x) else "finite"
                                            for x in self.learned_threshold_by_tau],
            "validation_stats": self.validation_stats,
            "development_objective": float(self.development_objective),
            "validation_objective": float(self.validation_objective),
        }


class GRACE(_GRACECoreBase):
    """GRACE estimator.

    GRACE means *Gated Reliability and Cost-Aware Calibration for Early
    Decisions*.  At each admissible prefix, the estimator (i) obtains a
    calibrated class-probability vector, (ii) evaluates reliability using
    current/past-prefix information, (iii) selects the Bayes action under
    ``L(a,k)``, and (iv) applies a conjunctive or learned gated-acceptance
    rule.  Delay and observation costs are used for development/evaluation
    objectives and do not alter the definition of current-prefix action risk.

    ``gated_acceptance_family='conjunctive'`` specifies GRACE-C. For the learned
    family, ``learned_history_mode='soft'`` specifies GRACE-LS and ``'eligible'``
    specifies GRACE-LH. The LH history guard applies only when action stability
    is an active feature. Learned variants never force action agreement.

    ``classes_`` is the frozen LabelEncoder order and determines exact-risk
    ties. ``cost_class_order`` specifies the supplied matrix axes only. For
    string labels BUY, HOLD, SELL, the fitted order is BUY, HOLD, SELL even if
    the supplied matrix is ordered SELL, HOLD, BUY; its axes are permuted.
    """

    THEORY_FEATURES = (
        "pmax",
        "margin",
        "entropy",
        "action_stability",
        "risk_min",
        "risk_margin",
    )
    FEATURE_DIRECTIONS = {
        "pmax": "high",
        "margin": "high",
        "entropy": "low",
        "action_stability": "high",
        "risk_min": "low",
        "risk_margin": "high",
        "reliability_score": "high",
    }

    def __init__(
        self,
        prefix_model: Optional[BaseEstimator] = None,
        gated_acceptance_model: Optional[BaseEstimator] = None,
        reliability_model: Optional[BaseEstimator] = None,
        prefix_transformer: Optional[Callable[[ArrayLike, int], np.ndarray]] = None,
        transformer_input: str = "auto",
        decision_points: Optional[Sequence[float]] = None,
        calibration_method: str = "sigmoid",
        fit_mode: str = "crossfit",
        crossfit_folds: int = 3,
        crossfit_inner_folds: int = 3,
        crossfit_purge: int = 0,
        crossfit_embargo: int = 0,
        calibration_size: float = 0.20,
        reliability_size: float = 0.25,
        gated_acceptance_size: float = 0.30,
        gated_acceptance_validation_fraction: float = 0.50,
        split_strategy: str = "stratified",
        gated_acceptance_family: str = "conjunctive",
        gated_acceptance_features: Sequence[str] = THEORY_FEATURES,
        action_stability_window: int = 2,
        gated_acceptance_quantiles: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
        gated_acceptance_coordinate_passes: int = 2,
        learned_acceptance_thresholds: Sequence[float] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
        learned_acceptance_crossfit_folds: int = 3,
        reliability_mode: str = "fixed",
        reliability_weights: Optional[Union[ReliabilityWeights, Mapping[str, float]]] = None,
        reliability_crossfit_folds: int = 3,
        alpha_delay: float = 0.10,
        alpha_observation: float = 0.0,
        misclassification_cost: Optional[np.ndarray] = None,
        cost_class_order: Optional[Sequence[Any]] = None,
        delay_cost_fn: Optional[DelayCostFunction] = None,
        observation_cost_fn: Optional[DelayCostFunction] = None,
        cost_scenarios: Optional[np.ndarray] = None,
        cost_scenario_weights: Optional[np.ndarray] = None,
        delay_scenario_multipliers: Optional[np.ndarray] = None,
        robust_cost_mode: str = "nominal",
        cost_cvar_alpha: float = 0.90,
        drift_psi_warning: float = 0.10,
        drift_psi_alarm: float = 0.25,
        auto_safe_mode_on_shift: bool = True,
        force_terminal_full: bool = True,
        random_state: int = 42,
        learned_history_mode: str = "soft",
    ):
        self.prefix_model = prefix_model
        self.gated_acceptance_model = gated_acceptance_model
        self.reliability_model = reliability_model
        self.prefix_transformer = prefix_transformer
        self.transformer_input = transformer_input
        self.decision_points = decision_points
        self.calibration_method = calibration_method
        self.fit_mode = fit_mode
        self.crossfit_folds = crossfit_folds
        self.crossfit_inner_folds = crossfit_inner_folds
        self.crossfit_purge = crossfit_purge
        self.crossfit_embargo = crossfit_embargo
        self.calibration_size = calibration_size
        self.reliability_size = reliability_size
        self.gated_acceptance_size = gated_acceptance_size
        self.gated_acceptance_validation_fraction = gated_acceptance_validation_fraction
        self.split_strategy = split_strategy
        self.gated_acceptance_family = gated_acceptance_family
        self.gated_acceptance_features = gated_acceptance_features
        self.action_stability_window = action_stability_window
        self.gated_acceptance_quantiles = gated_acceptance_quantiles
        self.gated_acceptance_coordinate_passes = gated_acceptance_coordinate_passes
        self.learned_acceptance_thresholds = learned_acceptance_thresholds
        self.learned_acceptance_crossfit_folds = learned_acceptance_crossfit_folds
        self.reliability_mode = reliability_mode
        self.reliability_weights = reliability_weights
        self.reliability_crossfit_folds = reliability_crossfit_folds
        self.alpha_delay = alpha_delay
        self.alpha_observation = alpha_observation
        self.misclassification_cost = misclassification_cost
        self.cost_class_order = cost_class_order
        self.delay_cost_fn = delay_cost_fn
        self.observation_cost_fn = observation_cost_fn
        self.cost_scenarios = cost_scenarios
        self.cost_scenario_weights = cost_scenario_weights
        self.delay_scenario_multipliers = delay_scenario_multipliers
        self.robust_cost_mode = robust_cost_mode
        self.cost_cvar_alpha = cost_cvar_alpha
        self.drift_psi_warning = drift_psi_warning
        self.drift_psi_alarm = drift_psi_alarm
        self.auto_safe_mode_on_shift = auto_safe_mode_on_shift
        self.force_terminal_full = force_terminal_full
        self.random_state = random_state
        self.learned_history_mode = learned_history_mode


    def _validate_hyperparameters(self) -> None:
        for name, minimum in (
            ("action_stability_window", 1), ("crossfit_folds", 2),
            ("crossfit_inner_folds", 2), ("reliability_crossfit_folds", 2),
            ("learned_acceptance_crossfit_folds", 2), ("crossfit_purge", 0),
            ("crossfit_embargo", 0), ("gated_acceptance_coordinate_passes", 1),
            ("random_state", 0),
        ):
            _integer_control(getattr(self, name), name, minimum)
        if int(self.random_state) > np.iinfo(np.uint32).max:
            raise ValueError("random_state must lie in [0, 2**32 - 1].")
        for name in ("auto_safe_mode_on_shift", "force_terminal_full"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean.")
        for name in ("calibration_size", "reliability_size", "gated_acceptance_size",
                     "gated_acceptance_validation_fraction"):
            if not 0.0 < _finite_scalar(getattr(self, name), name) < 1.0:
                raise ValueError(f"{name} must lie strictly in (0,1).")
        if self.fit_mode == "holdout" and (
            float(self.calibration_size) + float(self.reliability_size)
            + float(self.gated_acceptance_size) >= 1.0
        ):
            raise ValueError("Holdout calibration, reliability and acceptance sizes must sum to less than one.")
        for name in ("alpha_delay", "alpha_observation", "drift_psi_warning", "drift_psi_alarm"):
            if _finite_scalar(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 < _finite_scalar(self.cost_cvar_alpha, "cost_cvar_alpha") < 1.0:
            raise ValueError("cost_cvar_alpha must lie strictly in (0,1).")
        for name in ("delay_cost_fn", "observation_cost_fn", "prefix_transformer"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError(f"{name} must be callable or None.")
        if self.calibration_method not in {"sigmoid", "isotonic", "temperature", "none"}:
            raise ValueError("calibration_method must be 'sigmoid', 'isotonic', 'temperature', or 'none'.")
        if self.fit_mode not in {"crossfit", "holdout"}:
            raise ValueError("fit_mode must be 'crossfit' or 'holdout'.")
        if self.split_strategy not in {"stratified", "chronological", "group"}:
            raise ValueError("split_strategy must be stratified, chronological, or group.")
        if self.transformer_input not in {"auto", "masked", "legacy"}:
            raise ValueError("transformer_input must be auto, masked, or legacy.")
        if self.gated_acceptance_family not in {"conjunctive", "learned", "auto"}:
            raise ValueError("gated_acceptance_family must be conjunctive, learned, or auto.")
        if self.learned_history_mode not in {"soft", "eligible"}:
            raise ValueError("learned_history_mode must be 'soft' (GRACE-LS) or 'eligible' (GRACE-LH).")
        features = tuple(str(value) for value in self.gated_acceptance_features)
        allowed = set(self.THEORY_FEATURES) | {"reliability_score"}
        if not features or len(set(features)) != len(features) or not set(features) <= allowed:
            raise ValueError(f"gated_acceptance_features must be a non-empty subset of {sorted(allowed)}.")
        if int(self.action_stability_window) < 1:
            raise ValueError("action_stability_window must be positive.")
        if int(self.crossfit_folds) < 2 or int(self.crossfit_inner_folds) < 2:
            raise ValueError("crossfit_folds and crossfit_inner_folds must be at least two.")
        if int(self.reliability_crossfit_folds) < 2 or int(self.learned_acceptance_crossfit_folds) < 2:
            raise ValueError("Reliability and learned-acceptance cross-fit folds must be at least two.")
        if int(self.crossfit_purge) < 0 or int(self.crossfit_embargo) < 0:
            raise ValueError("crossfit_purge and crossfit_embargo must be non-negative.")
        if not 0.0 < float(self.gated_acceptance_size) < 1.0:
            raise ValueError("gated_acceptance_size must lie in (0,1).")
        if not 0.0 < float(self.gated_acceptance_validation_fraction) < 1.0:
            raise ValueError("gated_acceptance_validation_fraction must lie in (0,1).")
        quantiles = np.asarray(list(self.gated_acceptance_quantiles), dtype=float)
        if quantiles.ndim != 1 or not len(quantiles) or np.any(~np.isfinite(quantiles)) or np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
            raise ValueError("gated_acceptance_quantiles must contain finite values strictly in (0,1).")
        thresholds = np.asarray(list(self.learned_acceptance_thresholds), dtype=float)
        if thresholds.ndim != 1 or not len(thresholds) or np.any(~np.isfinite(thresholds)) or np.any((thresholds < 0.0) | (thresholds > 1.0)):
            raise ValueError("learned_acceptance_thresholds must lie in [0,1].")
        if int(self.gated_acceptance_coordinate_passes) < 1:
            raise ValueError("gated_acceptance_coordinate_passes must be positive.")
        if self.reliability_mode not in {"fixed", "learned"}:
            raise ValueError("reliability_mode must be fixed or learned.")
        if float(self.alpha_delay) < 0.0 or float(self.alpha_observation) < 0.0:
            raise ValueError("alpha_delay and alpha_observation must be non-negative.")
        if not 0.0 <= float(self.drift_psi_warning) <= float(self.drift_psi_alarm):
            raise ValueError("Require 0 <= drift_psi_warning <= drift_psi_alarm.")
        if self.split_strategy == "group" and self.fit_mode == "holdout":
            raise ValueError("group splitting is supported with fit_mode='crossfit'.")

    def _training_partition_error(self, cause: ValueError, y: np.ndarray) -> CrossfitFeasibilityError:
        """Explain an infeasible role allocation without relaxing separation."""
        diagnostics = {
            "stage": "training_role_allocation", "fit_mode": self.fit_mode,
            "strategy": self.split_strategy, "training_rows": int(len(y)),
            "class_counts": {str(c): int(np.sum(y == k)) for k, c in enumerate(self.classes_)},
            "purge_rows": int(self.crossfit_purge), "embargo_rows": int(self.crossfit_embargo),
            "gated_acceptance_size": float(self.gated_acceptance_size),
            "gated_acceptance_validation_fraction": float(self.gated_acceptance_validation_fraction),
            "calibration_size": float(self.calibration_size), "reliability_size": float(self.reliability_size),
            "reason": str(cause), "prefix_classifiers_fitted": False,
            "remedies": [
                "Provide more historical training episodes before fitting the declared temporal roles.",
                "Prespecify role shares, lookback and horizon; retain the required information gap.",
                "Do not move external test episodes into training or manufacture missing classes.",
            ],
        }
        return CrossfitFeasibilityError(str(cause), diagnostics)

    def _validate_custom_holdout_partitions(
        self,
        split_indices: Mapping[str, Sequence[int]],
        y: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Validate public holdout split indices with fail-closed isolation rules."""
        required = (
            "prefix_prediction",
            "calibration",
            "reliability",
            "gated_acceptance_development",
            "gated_acceptance_validation",
        )
        if set(split_indices) != set(required):
            raise ValueError(f"split_indices must have exactly roles {sorted(required)}.")
        n = int(len(y))
        partitions: Dict[str, np.ndarray] = {}
        for role in required:
            raw = np.asarray(split_indices[role])
            if raw.ndim != 1 or raw.size == 0:
                raise ValueError(f"split_indices[{role!r}] must be a non-empty one-dimensional array.")
            if not np.issubdtype(raw.dtype, np.integer):
                # Reject floating-point values even if numerically integral: the public contract is integer indices.
                raise ValueError(f"split_indices[{role!r}] must contain integer indices only.")
            rows = raw.astype(int, copy=False)
            if np.any(rows < 0) or np.any(rows >= n):
                raise ValueError(f"split_indices[{role!r}] contains an out-of-range index for n={n}.")
            if len(np.unique(rows)) != len(rows):
                raise ValueError(f"split_indices[{role!r}] contains duplicate indices.")
            partitions[role] = np.sort(rows)

        owner: Dict[int, str] = {}
        for role in required:
            for row in partitions[role].tolist():
                if row in owner:
                    raise ValueError(
                        f"split_indices roles overlap at row {row}: {owner[row]!r} and {role!r}."
                    )
                owner[row] = role
        missing = sorted(set(range(n)).difference(owner))
        if missing:
            preview = missing[:10]
            raise ValueError(
                "Custom split_indices must completely disposition every training row exactly once; "
                f"unassigned rows include {preview}."
            )

        for role in required:
            if not self._split_has_all_classes(y[partitions[role]]):
                raise ValueError(
                    f"The custom {role} subset is missing at least one outcome class; "
                    "increase data or revise the split."
                )

        if groups is not None:
            seen_groups: Dict[Any, str] = {}
            group_array = np.asarray(groups, dtype=object)
            for role in required:
                for group in np.unique(group_array[partitions[role]]).tolist():
                    if group in seen_groups:
                        raise ValueError(
                            "Custom split_indices violate group disjointness: group "
                            f"{group!r} appears in {seen_groups[group]!r} and {role!r}."
                        )
                    seen_groups[group] = role

        if self.split_strategy == "chronological":
            if self._temporal_gap() > 0:
                raise ValueError(
                    "Custom chronological split_indices cannot encode discarded purge/embargo rows. "
                    "Set crossfit_purge=crossfit_embargo=0 or use GRACE-generated chronological partitions."
                )
            previous_max = -1
            for role in required:
                rows = partitions[role]
                if int(rows[0]) <= previous_max:
                    raise ValueError(
                        "Custom chronological split_indices violate role ordering; rows must follow "
                        "prefix_prediction -> calibration -> reliability -> gated development -> gated validation."
                    )
                previous_max = int(rows[-1])
        return partitions

    def _uses_fold_preprocessing(self) -> bool:
        state = getattr(self, "preprocessing_state_", None)
        return bool(
            state is not None
            and getattr(state, "uses_cross_episode_parameters_for_model", False)
        )

    @staticmethod
    def _batch_as_samples(batch: MaskedSeriesBatch) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        for i, length in enumerate(batch.lengths.astype(int)):
            values = np.asarray(batch.values[i, :length], dtype=float).copy()
            mask = np.asarray(batch.observation_mask[i, :length], dtype=bool)
            values[~mask] = np.nan
            samples.append(values)
        return samples

    def _new_fold_preprocessor(self) -> Any:
        template = getattr(self, "preprocessing_state_", None)
        if template is None:
            raise RuntimeError("No preprocessing template is available.")
        return type(template)(template.settings, tuple(template.channel_names))

    def _apply_preprocessor(self, batch: MaskedSeriesBatch, state: Any) -> MaskedSeriesBatch:
        if state is None:
            return batch
        result = state.transform(self._batch_as_samples(batch))
        transformed = result.values
        if bool(getattr(state.settings, "add_missing_indicators", False)):
            values_list = transformed if isinstance(transformed, list) else list(np.asarray(transformed))
            indicator_list = (
                result.missing_indicators
                if isinstance(result.missing_indicators, list)
                else list(np.asarray(result.missing_indicators))
            )
            transformed = [
                np.concatenate([np.asarray(values, dtype=float), np.asarray(indicators, dtype=float)], axis=1)
                for values, indicators in zip(values_list, indicator_list)
            ]
        processed = _canonicalize_X(transformed)
        processed = processed.align(batch.shape[1], processed.shape[2])
        # Rebuilding a batch must retain its observation clock. Replacing an
        # irregular clock by row numbers changes timestamp-aware slopes even
        # when preprocessing leaves every value unchanged.
        for sample in range(batch.shape[0]):
            observed_times = batch.timestamps[sample, processed.time_mask[sample]]
            if (not np.isfinite(observed_times).all()
                    or np.any(np.diff(observed_times) <= 0.0)):
                raise ValueError(
                    f"Stateful preprocessing activates time points with invalid timestamps in sample {sample}. "
                    "Supply finite, strictly increasing timestamps for every observed or imputed time point."
                )
        return MaskedSeriesBatch(
            values=processed.values,
            observation_mask=processed.observation_mask,
            timestamps=batch.timestamps.copy(),
            original_ndim=batch.original_ndim,
        )

    def _inner_calibration_partition(
        self,
        rows: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return leakage-controlled inner model/calibration rows.

        ``rows`` index the current development set.  The function is fail-closed:
        every returned subset must contain all classes, group mode is entity-disjoint,
        and chronological mode inserts the configured purge/embargo gap.
        """
        rows = np.asarray(rows, dtype=int)
        if rows.ndim != 1 or len(rows) < max(4, 2 * self.n_classes_):
            raise ValueError("Too few rows for an inner calibration partition.")

        if self.split_strategy == "group":
            if groups is None:
                raise ValueError("groups are required for group-aware inner calibration.")
            group_array = np.asarray(groups)
            local_groups = group_array[rows]
            local_y = y[rows]
            class_group_counts = [
                len(np.unique(local_groups[local_y == cls])) for cls in range(self.n_classes_)
            ]
            folds = min(int(self.crossfit_inner_folds), len(np.unique(local_groups)), min(class_group_counts))
            if folds < 2:
                raise ValueError("Inner group calibration needs at least two groups per class.")
            splitter = StratifiedGroupKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
            for local_fit, local_cal in splitter.split(np.zeros(len(rows)), local_y, local_groups):
                fit_rows, cal_rows = rows[local_fit], rows[local_cal]
                if not self._split_has_all_classes(y[fit_rows]) or not self._split_has_all_classes(y[cal_rows]):
                    continue
                if set(group_array[fit_rows]).intersection(set(group_array[cal_rows])):
                    continue
                return fit_rows, cal_rows, {
                    "strategy": "stratified_group",
                    "gap_rows": 0,
                    "fit_group_count": int(len(np.unique(group_array[fit_rows]))),
                    "calibration_group_count": int(len(np.unique(group_array[cal_rows]))),
                }
            raise ValueError("Could not form a class-feasible group-disjoint inner calibration split.")

        if self.split_strategy == "chronological":
            ordered = np.sort(rows)
            gap = int(self._temporal_gap())
            n = len(ordered)
            # Allocate the usable rows, excluding the explicitly discarded gap.
            # A fixed percentage band of the original rows can exclude every
            # valid cut when the gap is large, despite a feasible nested design.
            preferred = int(round(0.80 * (n - gap)))
            candidates = sorted(
                range(self.n_classes_, n - gap - self.n_classes_ + 1),
                key=lambda value: (abs(value - preferred), value),
            )
            for cut in candidates:
                if cut < self.n_classes_ or n - cut - gap < self.n_classes_:
                    continue
                fit_rows = ordered[:cut]
                cal_rows = ordered[cut + gap :]
                audit_start = len(getattr(self, "temporal_split_audit_", []))
                try:
                    fit_rows, cal_rows = self._purge_temporal_pair(fit_rows, cal_rows, "inner_calibration")
                except ValueError:
                    continue
                if self._split_has_all_classes(y[fit_rows]) and self._split_has_all_classes(y[cal_rows]):
                    return fit_rows, cal_rows, {
                        "strategy": "purged_chronological",
                        "gap_rows": gap,
                        "discarded_gap_rows": ordered[cut : cut + gap].copy(),
                        "cut_position": int(cut),
                        "cut_search": "gap_aware_feasible_range",
                        "preferred_model_fraction_of_non_gap_rows": 0.80,
                    }
                # Rejected candidates are not executed training partitions.
                if hasattr(self, "temporal_split_audit_"):
                    del self.temporal_split_audit_[audit_start:]
            raise ValueError("Could not form a class-feasible purged chronological inner calibration split.")

        local_y = y[rows]
        counts = np.bincount(local_y, minlength=self.n_classes_)
        if np.min(counts) < 2:
            raise ValueError("Inner stratified calibration requires at least two rows per class.")
        n_cal = min(len(rows)-self.n_classes_, max(self.n_classes_, int(np.ceil(len(rows)/int(self.crossfit_inner_folds)))))
        train_fraction = 1 - n_cal / len(rows)
        local_fit, local_cal = train_test_split(
            np.arange(len(rows)),
            train_size=train_fraction,
            stratify=local_y,
            random_state=int(seed),
        )
        fit_rows, cal_rows = rows[local_fit], rows[local_cal]
        if not self._split_has_all_classes(y[fit_rows]) or not self._split_has_all_classes(y[cal_rows]):
            raise ValueError("Inner stratified calibration split lost an outcome class.")
        return fit_rows, cal_rows, {"strategy": "stratified", "gap_rows": 0}

    def _fit_crossfit_predictors(
        self,
        X: MaskedSeriesBatch,
        y: np.ndarray,
        template: BaseEstimator,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        stateful = self._uses_fold_preprocessing()

        n, M, K = len(y), len(self.tau_grid_), self.n_classes_
        oof = np.full((n, M, K), np.nan, dtype=float)
        minimum = max(6, K * 2)
        self.crossfit_diagnostics_ = {
            "stage": "outer_partition", "strategy": self.split_strategy,
            "predictor_development_rows": int(n), "prefix_count": int(M),
            "class_counts": {str(c): int(np.sum(y == k)) for k, c in
                             enumerate(getattr(self, "classes_", range(K)))},
            "requested_outer_folds": int(self.crossfit_folds),
            "purge_rows": int(self.crossfit_purge), "embargo_rows": int(self.crossfit_embargo),
            "minimum_complete_trajectories": minimum,
            "complete_trajectories": 0, "feasible_folds": 0, "folds": [],
            "remedies": [
                "Provide more historical training episodes with repeated occurrences of every class.",
                "Review the lookback, forecast horizon and reserved gate partitions before fixing the study protocol.",
                "Keep temporal purging and independent calibration; do not replace missing OOF predictions with training predictions.",
            ],
        }
        try:
            splits = self._outer_splits(y, groups)
        except ValueError as exc:
            raise CrossfitFeasibilityError(
                f"Cannot construct the requested cross-fitting folds: {exc}", self.crossfit_diagnostics_
            ) from exc
        exclusions = getattr(self, "_outer_split_exclusions_", [])
        self.crossfit_diagnostics_["folds"].extend(exclusions)
        fold_ids = getattr(self, "_outer_split_fold_ids_", []) or list(range(len(splits)))
        self.crossfit_diagnostics_["constructed_outer_folds"] = len(splits)
        self.crossfit_splits_ = [
            {"train": train.copy(), "validation": valid.copy()} for train, valid in splits
        ]
        self.base_models_ = []
        self.calibrators_ = []
        self.feature_dimensions_ = []
        self.fold_preprocessors_: List[Any] = []
        self.calibration_inner_splits_: List[Dict[str, Any]] = []

        # The final deployable preprocessing state is fitted only on the predictor-model
        # rows of an explicit inner partition.  The calibration rows therefore never
        # contribute to stateful preprocessing parameters.
        all_rows = np.arange(n, dtype=int)
        self.crossfit_diagnostics_["stage"] = "final_calibration_partition"
        try:
            final_fit, final_cal, final_audit = self._inner_calibration_partition(
                all_rows, y, groups, (int(self.random_state) + 701) % (2 ** 32)
            )
        except ValueError as exc:
            raise CrossfitFeasibilityError(
                f"Cannot form independent predictor/calibration subsets from {n} development rows: {exc}",
                self.crossfit_diagnostics_,
            ) from exc
        final_state = (self._new_fold_preprocessor().fit(
            [self._batch_as_samples(X.take(final_fit))],
            fit_partitions=["predictor_development_inner_model"],
        ) if stateful else None)
        if stateful:
            self.final_preprocessor_ = final_state
        final_fit_batch = self._apply_preprocessor(X.take(final_fit), final_state)
        final_cal_batch = self._apply_preprocessor(X.take(final_cal), final_state)
        self.final_calibration_split_ = {
            "model_train": final_fit.copy(),
            "calibration": final_cal.copy(),
            **final_audit,
        }

        fold_payloads: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MaskedSeriesBatch, MaskedSeriesBatch, MaskedSeriesBatch, Any]] = []
        for fold, (outer_train, outer_valid) in zip(fold_ids, splits):
            diagnostic = {"fold": int(fold), "outer_training_rows": int(len(outer_train)),
                          "outer_validation_rows": int(len(outer_valid)),
                          "outer_training_class_counts": {
                              str(c): int(np.sum(y[outer_train] == k)) for k, c in
                              enumerate(getattr(self, "classes_", range(K)))}}
            try:
                inner_fit, inner_cal, inner_audit = self._inner_calibration_partition(
                    outer_train, y, groups, (int(self.random_state) + 9000 + fold) % (2 ** 32)
                )
            except ValueError as exc:
                self.calibration_inner_splits_.append({"fold":fold,"status":"excluded_infeasible","reason":str(exc)})
                diagnostic.update(status="excluded_infeasible", reason=str(exc))
                self.crossfit_diagnostics_["folds"].append(diagnostic)
                continue
            diagnostic.update(status="usable", model_training_rows=int(len(inner_fit)),
                              calibration_rows=int(len(inner_cal)),
                              model_training_class_counts={str(c): int(np.sum(y[inner_fit] == k))
                                  for k, c in enumerate(getattr(self, "classes_", range(K)))},
                              calibration_class_counts={str(c): int(np.sum(y[inner_cal] == k))
                                  for k, c in enumerate(getattr(self, "classes_", range(K)))},
                              cut_search=inner_audit.get("cut_search", "not_applicable"))
            self.crossfit_diagnostics_["folds"].append(diagnostic)
            state = (self._new_fold_preprocessor().fit(
                [self._batch_as_samples(X.take(inner_fit))],
                fit_partitions=[f"crossfit_{fold}_inner_model"],
            ) if stateful else None)
            self.fold_preprocessors_.append(state)
            fit_batch = self._apply_preprocessor(X.take(inner_fit), state)
            cal_batch = self._apply_preprocessor(X.take(inner_cal), state)
            valid_batch = self._apply_preprocessor(X.take(outer_valid), state)
            self.calibration_inner_splits_.append(
                {
                    "fold": int(fold),
                    "outer_train": outer_train.copy(),
                    "outer_validation": outer_valid.copy(),
                    "model_train": inner_fit.copy(),
                    "calibration": inner_cal.copy(),
                    **inner_audit,
                }
            )
            fold_payloads.append((outer_train, outer_valid, inner_fit, inner_cal, fit_batch, cal_batch, valid_batch, state))

        planned = len(np.unique(np.concatenate([item[1] for item in fold_payloads]))) if fold_payloads else 0
        self.crossfit_diagnostics_["folds"].sort(key=lambda item: item["fold"])
        self.crossfit_diagnostics_.update(stage="fold_feasibility", feasible_folds=len(fold_payloads),
                                         planned_complete_trajectories=int(planned))
        if planned < minimum:
            raise CrossfitFeasibilityError(
                f"Cross-fitting cannot provide enough independent probability trajectories: "
                f"{planned} usable rows; at least {minimum} required. "
                f"Usable folds: {len(fold_payloads)}/{len(splits) + len(exclusions)}; development rows: {n}; "
                f"purge+embargo: {self._temporal_gap()}. "
                "Provide more historical training data and inspect the per-fold class/calibration diagnostics.",
                self.crossfit_diagnostics_,
            )

        for m, tau in enumerate(self.tau_grid_):
            final_fit_features = self._transform(final_fit_batch, int(tau))
            final_cal_features = self._transform(final_cal_batch, int(tau))
            final_base = clone(template).fit(final_fit_features, y[final_fit])
            final_calibrator = self._fit_prefit_calibrator(
                final_base, final_cal_features, y[final_cal]
            )
            self.base_models_.append(final_base)
            self.calibrators_.append(final_calibrator)
            self.feature_dimensions_.append(int(final_fit_features.shape[1]))

            for fold, payload in enumerate(fold_payloads):
                _, outer_valid, inner_fit, inner_cal, fit_batch, cal_batch, valid_batch, _ = payload
                fit_features = self._transform(fit_batch, int(tau))
                cal_features = self._transform(cal_batch, int(tau))
                valid_features = self._transform(valid_batch, int(tau))
                base = clone(template).fit(fit_features, y[inner_fit])
                calibrator = self._fit_prefit_calibrator(base, cal_features, y[inner_cal])
                oof[outer_valid, m] = self._ordered_predict_proba(calibrator, valid_features)

        available = np.all(np.isfinite(oof), axis=(1, 2))
        self.crossfit_diagnostics_.update(stage="probability_coverage", complete_trajectories=int(np.sum(available)),
                                         uncovered_rows=int(np.sum(~available)))
        if int(np.sum(available)) < minimum:
            raise CrossfitFeasibilityError(
                "Cross-fitting returned insufficient finite probability trajectories; inspect predictor outputs.",
                self.crossfit_diagnostics_,
            )
        self.fold_preprocessing_status_ = "refitted_within_each_internal_crossfit_fold" if stateful else "stateless_nested_calibration"
        self.inner_calibration_preprocessing_status_ = "inner_model_rows_only"
        return oof, available

    def _fit_holdout_predictors(
        self, X: MaskedSeriesBatch, y: np.ndarray, partitions: Mapping[str, np.ndarray],
        template: BaseEstimator,
    ) -> None:
        if not self._uses_fold_preprocessing():
            self.fold_preprocessing_status_ = "stateless_or_external_transform"
            return super()._fit_holdout_predictors(X, y, partitions, template)
        base_rows, cal_rows = partitions["prefix_prediction"], partitions["calibration"]
        state = self._new_fold_preprocessor().fit(
            [self._batch_as_samples(X.take(base_rows))], fit_partitions=["holdout_prefix_prediction"]
        )
        self.final_preprocessor_ = state
        base_batch = self._apply_preprocessor(X.take(base_rows), state)
        cal_batch = self._apply_preprocessor(X.take(cal_rows), state)
        self.base_models_ = []
        self.calibrators_ = []
        self.feature_dimensions_ = []
        for tau in self.tau_grid_:
            base_features = self._transform(base_batch, int(tau))
            cal_features = self._transform(cal_batch, int(tau))
            model = clone(template).fit(base_features, y[base_rows])
            calibrator = self._fit_prefit_calibrator(model, cal_features, y[cal_rows])
            self.base_models_.append(model)
            self.calibrators_.append(calibrator)
            self.feature_dimensions_.append(int(base_features.shape[1]))
        self.fold_preprocessing_status_ = "fit_on_holdout_prefix_prediction_only"

    def _probability_grid(self, X: ArrayLike) -> np.ndarray:
        batch = _canonicalize_X(X)
        if self._uses_fold_preprocessing():
            check_is_fitted(self, attributes=["final_preprocessor_"])
            batch = self._apply_preprocessor(batch, self.final_preprocessor_)
        return super()._probability_grid(batch)

    def _action_risk_geometry(self, p: np.ndarray, tau_fraction: float = 0.0) -> Tuple[int, float, float]:
        risks = np.asarray(self.cost_model_.expected_action_risks(p) if self.cost_model_.robust_mode == "nominal" else self.cost_model_.expected_action_costs(p, tau_fraction=tau_fraction), dtype=float)
        comparisons = self.cost_model_._action_comparison_costs(p, tau_fraction=tau_fraction)
        order = np.argsort(comparisons, kind="mergesort")
        action = int(order[0])
        risk_min = float(risks[order[0]])
        risk_margin = float(comparisons[order[1]] - comparisons[order[0]]) if len(order) > 1 else float("inf")
        return action, risk_min, risk_margin

    def _theory_feature_grid(
        self,
        probabilities: np.ndarray,
        fractions: np.ndarray,
        valid: np.ndarray,
        reliability_score_grid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, float]]]]:
        n, M, _ = probabilities.shape
        actions = np.zeros((n, M), dtype=int)
        feature_names = list(self.THEORY_FEATURES)
        # The first pass constructs the four inputs needed by the reliability
        # model. Its output exists only on the second pass, so it cannot be
        # required as an input to its own construction.
        if reliability_score_grid is not None and "reliability_score" in tuple(self.gated_acceptance_features):
            feature_names.append("reliability_score")
        features = np.zeros((n, M, len(feature_names)), dtype=float)
        details: List[List[Dict[str, float]]] = []
        v = int(self.action_stability_window)
        for i in range(n):
            row: List[Dict[str, float]] = []
            action_history: List[int] = []
            for m in range(M):
                p = _probability_vector(probabilities[i, m])
                ordered = np.sort(p)[::-1]
                positive = p[p > 0.0]
                entropy = float(-np.sum(positive * np.log(positive)) / np.log(self.n_classes_))
                action, risk_min, risk_margin = self._action_risk_geometry(p, float(fractions[i, m]))
                actions[i, m] = action
                if valid[i, m]:
                    action_history.append(action)
                stable = 0.0
                if valid[i, m] and len(action_history) >= v:
                    recent = action_history[-v:]
                    stable = float(all(value == recent[0] for value in recent))
                info = {
                    "pmax": float(ordered[0]),
                    "margin": float(ordered[0] - ordered[1]),
                    "entropy": entropy,
                    "entropy_normalized": entropy,
                    "action_stability": stable,
                    "available_prefix_count": int(len(action_history)),
                    "risk_min": risk_min,
                    "risk_margin": risk_margin,
                    "tau_fraction": float(fractions[i, m]),
                    "expected_total_cost": self.cost_model_.expected_total_cost(
                        p, tau_fraction=float(fractions[i, m])
                    ),
                }
                if reliability_score_grid is not None:
                    info["reliability_score"] = float(reliability_score_grid[i, m])
                features[i, m] = np.asarray([info[name] for name in feature_names], dtype=float)
                row.append(info)
            details.append(row)
        self.theory_feature_names_ = tuple(feature_names)
        return features, actions, details

    def _reliability_design(
        self, theory_features: np.ndarray
    ) -> np.ndarray:
        indices = [self.THEORY_FEATURES.index(name) for name in (
            "pmax", "margin", "entropy", "action_stability"
        )]
        return np.asarray(theory_features[:, :, indices], dtype=float)

    def _fixed_reliability(
        self, details: List[List[Dict[str, float]]], valid: np.ndarray
    ) -> np.ndarray:
        result = np.zeros(valid.shape, dtype=float)
        for i in range(valid.shape[0]):
            for m in range(valid.shape[1]):
                if valid[i, m]:
                    result[i, m] = reliability_score(
                        details[i][m], self.n_classes_, self.reliability_weights_
                    )
        return result

    def _fit_reliability(
        self,
        theory_features: np.ndarray,
        details: List[List[Dict[str, float]]],
        actions: np.ndarray,
        y: np.ndarray,
        valid: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit or compute GRACE reliability from four theoretical components."""
        design = self._reliability_design(theory_features)
        fixed = self._fixed_reliability(details, valid)
        if self.reliability_mode == "fixed":
            self.reliability_model_ = None
            self.reliability_training_summary_ = {
                "mode": "fixed",
                "features": ["pmax", "margin", "entropy", "action_stability"],
                "rows": int(np.sum(valid)),
                "oof_coverage": 1.0,
            }
            return fixed

        template = (
            self.reliability_model
            if self.reliability_model is not None
            else LogisticRegression(max_iter=2000, random_state=int(self.random_state))
        )
        target = actions == y[:, None]
        oof = np.full(valid.shape, np.nan, dtype=float)
        splits = self._sample_level_splits(
            y, int(self.reliability_crossfit_folds), 200, groups
        )
        for train, held in splits:
            train_mask = valid[train]
            X_train = design[train][train_mask]
            y_train = target[train][train_mask].astype(int)
            if len(X_train) == 0:
                continue
            estimator = self._fit_binary(template, X_train, y_train)
            for sample in held:
                mask = valid[sample]
                if np.any(mask):
                    oof[sample, mask] = self._positive_probability(
                        estimator, design[sample, mask]
                    )
        observed_oof = valid & np.isfinite(oof)
        missing = valid & ~np.isfinite(oof)
        oof[missing] = fixed[missing]
        final_X = design[valid]
        final_y = target[valid].astype(int)
        self.reliability_model_ = self._fit_binary(template, final_X, final_y)
        self.reliability_training_summary_ = {
            "mode": "learned",
            "features": ["pmax", "margin", "entropy", "action_stability"],
            "rows": int(len(final_y)),
            "correct_action_rate": float(np.mean(final_y)),
            "oof_coverage": float(np.sum(observed_oof) / max(1, int(np.sum(valid)))),
            "fallback_rows": int(np.sum(missing)),
        }
        self.reliability_oof_ = oof.copy()
        return np.clip(oof, 0.0, 1.0)

    def _predict_reliability(
        self,
        theory_features: np.ndarray,
        details: List[List[Dict[str, float]]],
        valid: np.ndarray,
    ) -> np.ndarray:
        design = self._reliability_design(theory_features)
        if self.reliability_mode == "fixed" or self.reliability_model_ is None:
            return self._fixed_reliability(details, valid)
        result = np.zeros(valid.shape, dtype=float)
        if np.any(valid):
            result[valid] = self._positive_probability(
                self.reliability_model_, design[valid]
            )
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _attach_reliability(
        details: List[List[Dict[str, float]]], reliability: np.ndarray
    ) -> None:
        for i, row in enumerate(details):
            for m, info in enumerate(row):
                info["reliability"] = float(reliability[i, m])

    def _set_shift_reference(
        self, details: List[List[Dict[str, float]]], valid: np.ndarray
    ) -> None:
        self.shift_reference_: Dict[str, np.ndarray] = {}
        metrics = (
            "pmax", "margin", "entropy", "action_stability",
            "risk_min", "risk_margin", "reliability", "expected_total_cost"
        )
        for m, tau in enumerate(self.tau_grid_):
            rows = np.flatnonzero(valid[:, m])
            for metric in metrics:
                self.shift_reference_[f"{int(tau)}:{metric}"] = np.asarray(
                    [details[i][m][metric] for i in rows], dtype=float
                )

    def monitor_shift(
        self,
        X: ArrayLike,
        y: Optional[Sequence[Any]] = None,
        *,
        n_bins: int = 10,
        activate_safe_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Monitor current-prefix GRACE decision features for distribution shift."""
        check_is_fitted(self, attributes=["shift_reference_", "calibrators_"])
        batch = self._validate_prediction_input(X)
        _, fractions, _, valid = self._decision_geometry(batch)
        probabilities = self._probability_grid(batch)
        core_features, _, core_details = self._theory_feature_grid(
            probabilities, fractions, valid
        )
        reliability = self._predict_reliability(core_features, core_details, valid)
        _, _, details = self._theory_feature_grid(
            probabilities, fractions, valid, reliability
        )
        self._attach_reliability(details, reliability)

        metrics = (
            "pmax", "margin", "entropy", "action_stability",
            "risk_min", "risk_margin", "reliability", "expected_total_cost"
        )
        comparisons: List[Dict[str, Any]] = []
        for m, tau in enumerate(self.tau_grid_):
            rows = np.flatnonzero(valid[:, m])
            for metric in metrics:
                reference = self.shift_reference_.get(f"{int(tau)}:{metric}", np.empty(0))
                current = np.asarray([details[i][m][metric] for i in rows], dtype=float)
                if len(reference) and len(current):
                    ks = ks_2samp(reference, current, alternative="two-sided", method="auto")
                    comparisons.append({
                        "decision_point": int(tau),
                        "feature": metric,
                        "psi": self._population_stability_index(reference, current, n_bins),
                        "ks_statistic": float(ks.statistic),
                        "ks_pvalue": float(ks.pvalue),
                    })
        max_psi = max((row["psi"] for row in comparisons), default=0.0)
        mean_psi = float(np.mean([row["psi"] for row in comparisons])) if comparisons else 0.0
        status = (
            "alarm" if max_psi >= float(self.drift_psi_alarm)
            else "warning" if max_psi >= float(self.drift_psi_warning)
            else "stable"
        )
        report: Dict[str, Any] = {
            "status": status,
            "max_psi": float(max_psi),
            "mean_psi": mean_psi,
            "comparisons": comparisons,
            "safe_mode_before": bool(self.safe_mode_),
        }
        if y is not None:
            report["calibration_by_prefix"] = evaluate_calibration_by_prefix(
                self, batch, y, n_bins=n_bins
            )
        activate = (
            bool(self.auto_safe_mode_on_shift)
            if activate_safe_mode is None else bool(activate_safe_mode)
        )
        if status == "alarm" and activate:
            self.set_safe_mode(True, "distribution_shift_alarm")
        report["safe_mode_after"] = bool(self.safe_mode_)
        return report

    def _realized_objective(
        self,
        accept: np.ndarray,
        actions: np.ndarray,
        y: np.ndarray,
        fractions: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        n = accept.shape[0]
        chosen = np.zeros(n, dtype=int)
        costs = np.zeros(n, dtype=float)
        for i in range(n):
            valid_idx = np.flatnonzero(valid[i])
            terminal = int(valid_idx[-1])
            candidates = valid_idx[(accept[i, valid_idx]) & (valid_idx < terminal)]
            m = int(candidates[0]) if len(candidates) else terminal
            chosen[i] = m
            costs[i] = self.cost_model_.total_cost(
                int(y[i]), int(actions[i, m]), tau_fraction=float(fractions[i, m])
            )
        return float(np.mean(costs)), chosen

    def _threshold_acceptance(
        self,
        details: List[List[Dict[str, float]]],
        valid: np.ndarray,
        thresholds: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        n, M = valid.shape
        if bool(getattr(self, "conjunctive_terminal_only_", False)):
            return np.zeros((n, M), dtype=bool)
        enabled = np.asarray(
            getattr(self, "conjunctive_enabled_by_tau_", np.ones(M, dtype=bool)),
            dtype=bool,
        )
        accept = np.broadcast_to(enabled, (n, M)).copy()
        for i in range(n):
            for m in range(M):
                if not valid[i, m] or not enabled[m]:
                    accept[i, m] = False
                    continue
                for feature in self.gated_acceptance_features:
                    value = float(details[i][m][feature])
                    threshold = float(thresholds[feature][m])
                    direction = self.FEATURE_DIRECTIONS[feature]
                    if direction == "high" and value < threshold:
                        accept[i, m] = False
                        break
                    if direction == "low" and value > threshold:
                        accept[i, m] = False
                        break
        return accept

    def _threshold_candidates(self, feature: str, values: np.ndarray) -> np.ndarray:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if feature == "action_stability":
            return np.asarray([1.0])
        direction = self.FEATURE_DIRECTIONS[feature]
        permissive = -np.inf if direction == "high" else np.inf
        if len(finite) == 0:
            return np.asarray([permissive])
        quantiles = np.quantile(finite, np.asarray(self.gated_acceptance_quantiles, dtype=float))
        restrictive = np.inf if direction == "high" else -np.inf
        return np.unique(np.concatenate([[permissive, restrictive], np.asarray(quantiles, dtype=float)]))

    def _canonicalize_conjunctive_schedule(
        self, thresholds: Mapping[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compile search sentinels into finite thresholds and enable flags.

        Search may use infinities as always-pass/always-reject sentinels.
        Deployed GRACE-C instead uses finite coordinate thresholds and an
        explicit nonterminal enable flag. Every valid indicator is finite
        and non-negative; the largest finite float is a neutral upper bound
        without changing decisions through rounding near a domain boundary.
        """
        M = len(self.tau_grid_)
        enabled = np.ones(M, dtype=bool)
        finite_thresholds: Dict[str, np.ndarray] = {}
        for feature, values in thresholds.items():
            values = np.asarray(values, dtype=float)
            high = self.FEATURE_DIRECTIONS[feature] == "high"
            restrictive = np.isposinf(values) if high else np.isneginf(values)
            enabled[:-1] &= ~restrictive[:-1]
            neutral = 0.0 if high else np.finfo(float).max
            finite_thresholds[feature] = np.where(np.isfinite(values), values, neutral)
            if feature == "action_stability":
                finite_thresholds[feature][:] = 1.0
        if bool(getattr(self, "conjunctive_terminal_only_", False)):
            enabled[:-1] = False
        return finite_thresholds, enabled

    def _fit_conjunctive_acceptance(
        self,
        details: List[List[Dict[str, float]]],
        actions: np.ndarray,
        y: np.ndarray,
        fractions: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, np.ndarray]:
        M = valid.shape[1]
        self.conjunctive_terminal_only_ = False
        self.conjunctive_enabled_by_tau_ = np.ones(M, dtype=bool)
        thresholds: Dict[str, np.ndarray] = {}
        for feature in self.gated_acceptance_features:
            direction = self.FEATURE_DIRECTIONS[feature]
            initial = -np.inf if direction == "high" else np.inf
            thresholds[feature] = np.full(M, initial, dtype=float)
            if feature == "action_stability":
                thresholds[feature][:-1] = 1.0
        for _ in range(int(self.gated_acceptance_coordinate_passes)):
            for m in range(max(0, M - 1)):
                rows = np.flatnonzero(valid[:, m])
                for feature in self.gated_acceptance_features:
                    values = np.asarray([details[i][m][feature] for i in rows], dtype=float)
                    candidates = np.unique(np.r_[self._threshold_candidates(feature, values), thresholds[feature][m]])
                    best_value = float(thresholds[feature][m])
                    best_objective = float("inf")
                    best_fraction = float("inf")
                    for candidate in candidates:
                        previous = float(thresholds[feature][m])
                        thresholds[feature][m] = float(candidate)
                        accept = self._threshold_acceptance(details, valid, thresholds)
                        objective, chosen = self._realized_objective(
                            accept, actions, y, fractions, valid
                        )
                        mean_fraction = float(np.mean(fractions[np.arange(len(y)), chosen]))
                        key = (objective, mean_fraction)
                        if key < (best_objective, best_fraction):
                            best_objective, best_fraction = key
                            best_value = float(candidate)
                        thresholds[feature][m] = previous
                    thresholds[feature][m] = best_value
        accept = self._threshold_acceptance(details, valid, thresholds)
        objective, chosen = self._realized_objective(accept, actions, y, fractions, valid)
        terminal_objective, terminal_chosen = self._realized_objective(
            np.zeros_like(valid), actions, y, fractions, valid
        )
        current_key = (objective, float(np.mean(fractions[np.arange(len(y)), chosen])))
        terminal_key = (terminal_objective, float(np.mean(fractions[np.arange(len(y)), terminal_chosen])))
        if terminal_key < current_key:
            self.conjunctive_terminal_only_ = True
            objective, chosen = terminal_objective, terminal_chosen
        thresholds, self.conjunctive_enabled_by_tau_ = self._canonicalize_conjunctive_schedule(thresholds)
        return thresholds, objective, chosen

    def _fit_learned_acceptance(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        y: np.ndarray,
        fractions: np.ndarray,
        valid: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[BaseEstimator]], np.ndarray, float, np.ndarray]:
        n, M, _ = features.shape
        active_idx = [self.theory_feature_names_.index(name) for name in self.gated_acceptance_features]
        template = self.gated_acceptance_model if self.gated_acceptance_model is not None else LogisticRegression(
            max_iter=2000, random_state=(int(self.random_state) + 401) % (2 ** 32)
        )
        models: List[Optional[BaseEstimator]] = [None] * M
        oof_scores = np.full((n, M), -np.inf, dtype=float)
        for m in range(max(0, M - 1)):
            rows = np.flatnonzero(valid[:, m])
            X_m = features[rows, m][:, active_idx]
            target = (actions[rows, m] == y[rows]).astype(int)
            if len(rows) == 0:
                continue
            final_model = self._fit_binary(template, X_m, target)
            models[m] = final_model
            # Cross-fitted acceptance scores on Dg for threshold selection.
            local_groups = None if groups is None else groups[rows]
            active = self._active_temporal_rows_.copy()
            self._active_temporal_rows_ = active[rows]
            try:
                splits = self._sample_level_splits(
                    target, int(self.learned_acceptance_crossfit_folds), 500 + m, local_groups
                )
            except ValueError:
                splits = []
            finally:
                self._active_temporal_rows_ = active
            covered = np.zeros(len(rows), dtype=bool)
            for train_local, held_local in splits:
                estimator = self._fit_binary(template, X_m[train_local], target[train_local])
                oof_scores[rows[held_local], m] = self._positive_probability(
                    estimator, X_m[held_local]
                )
                covered[held_local] = True
            if np.any(~covered):
                # Uncovered rows cannot authorize acceptance, even at threshold zero.
                oof_scores[rows[~covered], m] = -np.inf
        oof_scores[~self._learned_history_eligibility(valid)] = -np.inf
        covered_by_prefix = np.isfinite(oof_scores).any(axis=0)
        thresholds = np.where(covered_by_prefix, 1.0, np.inf)
        thresholds[-1] = 1.0  # Terminal stopping is unconditional, not learned.
        candidates = np.unique(np.r_[np.asarray(self.learned_acceptance_thresholds, dtype=float), np.inf])
        for _ in range(int(self.gated_acceptance_coordinate_passes)):
            for m in range(max(0, M - 1)):
                if not covered_by_prefix[m]:
                    # A deployment gate cannot be enabled by a tie among
                    # thresholds that were never evaluated on an independent
                    # score. No OOF evidence means no learned early decision.
                    continue
                best = float(thresholds[m])
                best_key = (float("inf"), float("inf"))
                for candidate in np.unique(np.r_[candidates, thresholds[m]]):
                    old = float(thresholds[m])
                    thresholds[m] = float(candidate)
                    accept = (oof_scores >= thresholds[None, :]) & valid
                    objective, chosen = self._realized_objective(
                        accept, actions, y, fractions, valid
                    )
                    key = (
                        objective,
                        float(np.mean(fractions[np.arange(len(y)), chosen])),
                    )
                    if key < best_key:
                        best_key, best = key, float(candidate)
                    thresholds[m] = old
                thresholds[m] = best
        accept = (oof_scores >= thresholds[None, :]) & valid
        objective, chosen = self._realized_objective(accept, actions, y, fractions, valid)
        terminal_objective, terminal_chosen = self._realized_objective(
            np.zeros_like(valid), actions, y, fractions, valid
        )
        if (terminal_objective, float(np.mean(fractions[np.arange(n), terminal_chosen]))) < (
            objective, float(np.mean(fractions[np.arange(n), chosen]))
        ):
            thresholds[:-1] = np.inf
            objective, chosen = terminal_objective, terminal_chosen
        self.learned_acceptance_oof_summary_ = {
            "eligible_rows_by_prefix": self._learned_history_eligibility(valid).sum(axis=0).astype(int).tolist(),
            "finite_oof_scores_by_prefix": np.isfinite(oof_scores).sum(axis=0).astype(int).tolist(),
            "disabled_prefix_indices_no_oof": np.flatnonzero(~covered_by_prefix[:-1]).astype(int).tolist(),
            "uncovered_score_policy": "cannot_authorize_acceptance",
        }
        return models, thresholds, objective, chosen

    def _learned_history_eligibility(self, valid: np.ndarray) -> np.ndarray:
        """History eligibility for LS/LH, independent of actual action agreement.

        GRACE-LS treats stability as a numerical predictor. GRACE-LH requires
        at least ``action_stability_window`` available prefixes when stability
        is active. Removing stability removes this restriction in both modes.
        Neither learned mode requires the stability indicator itself to be one.
        """
        eligible = np.asarray(valid, dtype=bool).copy()
        if self.learned_history_mode == "eligible" and "action_stability" in self.gated_acceptance_features:
            eligible &= np.cumsum(valid, axis=1) >= int(self.action_stability_window)
        return eligible

    def _learned_acceptance_scores(
        self, features: np.ndarray, valid: np.ndarray, models: Sequence[Optional[BaseEstimator]]
    ) -> np.ndarray:
        n, M, _ = features.shape
        active_idx = [self.theory_feature_names_.index(name) for name in self.gated_acceptance_features]
        scores = np.full((n, M), -np.inf, dtype=float)
        for m in range(max(0, M - 1)):
            rows = np.flatnonzero(valid[:, m])
            if not len(rows) or models[m] is None:
                continue
            scores[rows, m] = self._positive_probability(
                models[m], features[rows, m][:, active_idx]
            )
        scores[~self._learned_history_eligibility(valid)] = -np.inf
        return scores

    def _gated_acceptance_validation_stats(
        self,
        family: str,
        accept: np.ndarray,
        actions: np.ndarray,
        y: np.ndarray,
        fractions: np.ndarray,
        valid: np.ndarray,
        enabled_by_tau: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        objective, _ = self._realized_objective(accept, actions, y, fractions, valid)
        stats: List[Dict[str, Any]] = []
        for m, tau in enumerate(self.tau_grid_):
            rows = np.flatnonzero(valid[:, m])
            accepted = rows[accept[rows, m]] if m < len(self.tau_grid_) - 1 else rows
            correct = int(np.sum(actions[accepted, m] == y[accepted])) if len(accepted) else 0
            stats.append(
                {
                    "tau": int(tau),
                    "family": family,
                    "enabled": bool(m < len(self.tau_grid_) - 1 and not (
                        family == "conjunctive" and getattr(self, "conjunctive_terminal_only_", False)
                    ) and (enabled_by_tau is None or enabled_by_tau[m])),
                    "evaluable": int(len(rows)),
                    "accepted": int(len(accepted)),
                    "accepted_action_accuracy": float(correct / len(accepted)) if len(accepted) else None,
                    "coverage": float(len(accepted) / max(1, len(rows))),
                }
            )
        return stats, objective

    def fit(
        self,
        X: ArrayLike,
        y: Sequence[Any],
        *,
        split_indices: Optional[Mapping[str, Sequence[int]]] = None,
        groups: Optional[Sequence[Any]] = None,
        episode_time: Optional[Sequence[Any]] = None,
        information_start: Optional[Sequence[Any]] = None,
        information_end: Optional[Sequence[Any]] = None,
    ) -> "GRACE":
        _fit_begin = time.perf_counter()
        # A failed refit must not leave an old deployable gate combined with
        # partially updated classes, dimensions or probability models.
        # ``preprocessing_state_`` is an explicit caller-supplied template.
        for name in tuple(vars(self)):
            if name.endswith("_") and name != "preprocessing_state_":
                delattr(self, name)
        self._validate_hyperparameters()
        batch = _canonicalize_X(X)
        n, self.T_, self.n_channels_in_ = batch.shape
        validated_labels = _validate_labels(y, "y")
        try:
            y_array = np.asarray(validated_labels.tolist())
            if not isinstance(validated_labels[0], str) and any(
                int(before) != int(after) for before, after in zip(validated_labels, y_array)
            ):
                raise ValueError("Numeric conversion changes a class identifier.")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Class identifiers cannot be represented without changing their values; "
                "use exact string identifiers for mixed large numeric labels."
            ) from exc
        if len(y_array) != n:
            raise ValueError("y must be one-dimensional and aligned with X.")
        groups_array = validate_groups(groups,n,"groups")
        if groups_array is not None and self.split_strategy != "group":
            raise ValueError("Groups require split_strategy=group.")
        self.temporal_metadata_ = validate_temporal(n,episode_time,information_start,information_end) if self.split_strategy == "chronological" else None
        self._active_temporal_rows_ = np.arange(n)
        self.temporal_split_audit_ = []
        if groups_array is not None and (groups_array.ndim != 1 or len(groups_array) != n):
            raise ValueError("groups must be one-dimensional and aligned with X.")
        if self.split_strategy == "group" and groups_array is None:
            raise ValueError("groups are required when split_strategy='group'.")

        self.label_encoder_ = LabelEncoder().fit(y_array)
        y_encoded = self.label_encoder_.transform(y_array).astype(int)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("GRACE requires at least two classes.")
        self.channel_names_ = batch.channel_names or tuple(
            f"channel_{index}" for index in range(self.n_channels_in_)
        )
        self.tau_grid_ = self._resolve_tau_grid(self.T_)
        self.prefix_transformer_ = default_prefix_transformer if self.prefix_transformer is None else self.prefix_transformer
        self.reliability_weights_ = self._resolve_weights()
        self.cost_matrix_, scenarios = self._validate_cost_configuration()
        self.cost_model_ = CostModel(
            alpha_delay=float(self.alpha_delay),
            alpha_observation=float(self.alpha_observation),
            tau_max=int(self.T_),
            misclassification_cost=self.cost_matrix_,
            delay_cost_fn=self.delay_cost_fn,
            observation_cost_fn=self.observation_cost_fn,
            cost_scenarios=scenarios,
            scenario_weights=self.cost_scenario_weights,
            delay_multipliers=self.delay_scenario_multipliers,
            robust_mode=self.robust_cost_mode,
            cvar_alpha=float(self.cost_cvar_alpha),
        )
        base_template = self.prefix_model if self.prefix_model is not None else LogisticRegression(
            max_iter=3000, random_state=int(self.random_state)
        )
        _, all_fractions, _, all_valid = self._decision_geometry(batch)
        self.cost_function_grid_audit_ = self.cost_model_.validate_time_cost_grid(
            np.unique(all_fractions[all_valid])
        )

        if self.fit_mode == "crossfit":
            if split_indices is not None:
                raise ValueError("Custom split_indices are supported only with fit_mode='holdout'.")
            try:
                reserved = self._isolate_temporal_roles(self._crossfit_partitions(y_encoded, groups_array))
            except ValueError as exc:
                raise self._training_partition_error(exc, y_encoded) from exc
            development = reserved["development"]
            self._active_temporal_rows_ = development
            self.calibration_split_index_scope_ = "predictor_development_local"
            _predictor_begin = time.perf_counter()
            _preparation_seconds = _predictor_begin - _fit_begin
            P_dev_oof, available = self._fit_crossfit_predictors(
                batch.take(development), y_encoded[development], base_template,
                None if groups_array is None else groups_array[development],
            )
            _predictor_end = time.perf_counter()
            self.crossfit_splits_ = [
                {"train": development[item["train"]], "validation": development[item["validation"]]}
                for item in self.crossfit_splits_
            ]
            eligible = np.flatnonzero(available)
            development_oof_rows = development[eligible]
            self.predictor_development_indices_ = development.copy()
            self.crossfit_oof_indices_ = development_oof_rows.copy()
            self.oof_probability_grid_ = P_dev_oof[eligible].copy()
            partitions = {
                "development": development,
                "gated_acceptance_development": reserved["gated_acceptance_development"],
                "gated_acceptance_validation": reserved["gated_acceptance_validation"],
            }
            # GRACE Reliability is trained only from OOF current-prefix trajectories.
            y_rel = y_encoded[development_oof_rows]
            valid_rel = all_valid[development_oof_rows]
            frac_rel = all_fractions[development_oof_rows]
            rel_features, actions_rel, rel_details = self._theory_feature_grid(
                self.oof_probability_grid_, frac_rel, valid_rel
            )
            self._active_temporal_rows_ = development_oof_rows
            self._fit_reliability(
                rel_features, rel_details, actions_rel, y_rel, valid_rel,
                None if groups_array is None else groups_array[development_oof_rows],
            )
        else:
            if split_indices is None:
                try:
                    partitions = self._isolate_temporal_roles(self._holdout_partitions(y_encoded))
                except ValueError as exc:
                    raise self._training_partition_error(exc, y_encoded) from exc
            else:
                partitions = self._validate_custom_holdout_partitions(
                    split_indices, y_encoded, groups_array
                )
            if split_indices is not None:
                clean = self._isolate_temporal_roles({k:v.copy() for k,v in partitions.items()})
                if any(not np.array_equal(clean[k],v) for k,v in partitions.items()):
                    raise ValueError("Custom temporal information intervals overlap.")
            _predictor_begin = time.perf_counter()
            _preparation_seconds = _predictor_begin - _fit_begin
            self._fit_holdout_predictors(batch, y_encoded, partitions, base_template)
            _predictor_end = time.perf_counter()
            P_all = self._probability_grid(batch)
            rel_rows = partitions["reliability"]
            self._active_temporal_rows_ = rel_rows
            y_rel = y_encoded[rel_rows]
            valid_rel = all_valid[rel_rows]
            frac_rel = all_fractions[rel_rows]
            rel_features, actions_rel, rel_details = self._theory_feature_grid(
                P_all[rel_rows], frac_rel, valid_rel
            )
            self._fit_reliability(
                rel_features, rel_details, actions_rel, y_rel, valid_rel,
                None if groups_array is None else groups_array[rel_rows],
            )
            self.predictor_development_indices_ = np.concatenate(
                [partitions["prefix_prediction"], partitions["calibration"]]
            )

        self.split_indices_ = {key: np.asarray(value, dtype=int).copy() for key, value in partitions.items()}
        self.split_summary_ = {key: int(len(value)) for key, value in self.split_indices_.items()}
        self.discarded_training_rows_ = np.setdiff1d(np.arange(n),np.unique(np.concatenate(list(self.split_indices_.values()))))
        self.split_summary_["discarded_training_rows"] = self.discarded_training_rows_.tolist()
        self.split_summary_["predictor_fit_rows"] = int(len(self.predictor_development_indices_))
        self.split_summary_["gated_acceptance_development_rows"] = int(
            len(self.split_indices_["gated_acceptance_development"])
        )
        self.split_summary_["gated_acceptance_validation_rows"] = int(
            len(self.split_indices_["gated_acceptance_validation"])
        )
        self.split_summary_["crossfit_purge_windows"] = int(self.crossfit_purge)
        self.split_summary_["crossfit_embargo_windows"] = int(self.crossfit_embargo)
        self.split_summary_["effective_temporal_gap_windows"] = int(self._temporal_gap())

        acceptance_data: Dict[str, Dict[str, Any]] = {}
        for role in ("gated_acceptance_development", "gated_acceptance_validation"):
            rows = self.split_indices_[role]
            P = self._probability_grid(batch.take(rows))
            valid = all_valid[rows]
            fractions = all_fractions[rows]
            core_features, _, core_details = self._theory_feature_grid(P, fractions, valid)
            reliability = self._predict_reliability(core_features, core_details, valid)
            F, A, details = self._theory_feature_grid(P, fractions, valid, reliability)
            self._attach_reliability(details, reliability)
            acceptance_data[role] = {
                "rows": rows, "P": P, "valid": valid, "fractions": fractions,
                "features": F, "actions": A, "details": details,
                "reliability": reliability, "y": y_encoded[rows],
                "groups": None if groups_array is None else groups_array[rows],
            }

        development_acceptance = acceptance_data["gated_acceptance_development"]
        validation_acceptance = acceptance_data["gated_acceptance_validation"]
        # Frozen development trajectories exposed only for protocol/baseline tuning.
        # They never contain external-test observations.
        self.protocol_development_ = {
            "rows": np.asarray(development_acceptance["rows"], dtype=int).copy(),
            "probabilities": np.asarray(development_acceptance["P"], dtype=float).copy(),
            "fractions": np.asarray(development_acceptance["fractions"], dtype=float).copy(),
            "valid": np.asarray(development_acceptance["valid"], dtype=bool).copy(),
            "y_encoded": np.asarray(development_acceptance["y"], dtype=int).copy(),
            "groups": (
                None
                if development_acceptance["groups"] is None
                else np.asarray(development_acceptance["groups"], dtype=object).copy()
            ),
        }
        self._active_temporal_rows_ = np.asarray(development_acceptance["rows"])
        candidate_results: Dict[str, Dict[str, Any]] = {}

        if self.gated_acceptance_family in {"conjunctive", "auto"}:
            thresholds, dev_obj, _ = self._fit_conjunctive_acceptance(
                development_acceptance["details"], development_acceptance["actions"], development_acceptance["y"], development_acceptance["fractions"], development_acceptance["valid"]
            )
            validation_accept = self._threshold_acceptance(validation_acceptance["details"], validation_acceptance["valid"], thresholds)
            stats, validation_obj = self._gated_acceptance_validation_stats(
                "conjunctive", validation_accept, validation_acceptance["actions"], validation_acceptance["y"],
                validation_acceptance["fractions"], validation_acceptance["valid"],
                enabled_by_tau=self.conjunctive_enabled_by_tau_,
            )
            candidate_results["conjunctive"] = {
                "thresholds": thresholds, "models": [None] * len(self.tau_grid_),
                "learned_thresholds": np.full(len(self.tau_grid_), 1.0),
                "development_objective": dev_obj, "validation_objective": validation_obj,
                "validation_stats": stats,
            }

        if self.gated_acceptance_family in {"learned", "auto"}:
            models, learned_thresholds, dev_obj, _ = self._fit_learned_acceptance(
                development_acceptance["features"], development_acceptance["actions"], development_acceptance["y"], development_acceptance["fractions"],
                development_acceptance["valid"], development_acceptance["groups"]
            )
            validation_scores = self._learned_acceptance_scores(validation_acceptance["features"], validation_acceptance["valid"], models)
            validation_accept = (validation_scores >= learned_thresholds[None, :]) & validation_acceptance["valid"]
            stats, validation_obj = self._gated_acceptance_validation_stats(
                "learned", validation_accept, validation_acceptance["actions"], validation_acceptance["y"],
                validation_acceptance["fractions"], validation_acceptance["valid"],
                enabled_by_tau=np.isfinite(learned_thresholds),
            )
            candidate_results["learned"] = {
                "thresholds": {}, "models": models,
                "learned_thresholds": learned_thresholds,
                "development_objective": dev_obj, "validation_objective": validation_obj,
                "validation_stats": stats,
            }

        selected_family = min(
            candidate_results,
            key=lambda key: (
                float(candidate_results[key]["development_objective"]),
                0 if key == "conjunctive" else 1,
            ),
        ) if self.gated_acceptance_family == "auto" else self.gated_acceptance_family
        selected = candidate_results[selected_family]
        self.gated_acceptance_models_ = selected["models"]
        self.selected_gated_acceptance_family_ = selected_family
        self.variant_ = "GRACE-C" if selected_family == "conjunctive" else (
            "GRACE-LH" if self.learned_history_mode == "eligible" else "GRACE-LS"
        )
        enabled = np.ones(len(self.tau_grid_), dtype=bool)
        if selected_family == "conjunctive":
            enabled = self.conjunctive_enabled_by_tau_.copy()
        if selected_family == "learned":
            enabled[:-1] = np.isfinite(selected["learned_thresholds"][:-1])
        if len(enabled):
            enabled[-1] = True
        self.gated_acceptance_schedule_ = GatedAcceptanceSchedule(
            tau_grid=self.tau_grid_.copy(),
            family=selected_family,
            active_features=tuple(str(x) for x in self.gated_acceptance_features),
            action_stability_window=int(self.action_stability_window),
            thresholds_by_feature={key: np.asarray(value, dtype=float) for key, value in selected["thresholds"].items()},
            learned_threshold_by_tau=np.asarray(selected["learned_thresholds"], dtype=float),
            enabled_by_tau=enabled,
            validation_stats=list(selected["validation_stats"]),
            development_objective=float(selected["development_objective"]),
            validation_objective=float(selected["validation_objective"]),
            learned_history_mode=self.learned_history_mode,
        )
        self.gated_acceptance_candidate_summary_ = {
            key: {
                "development_objective": float(value["development_objective"]),
                "validation_objective": float(value["validation_objective"]),
            }
            for key, value in candidate_results.items()
        }
        # Monitoring reference is frozen on gated-acceptance validation data.
        self._set_shift_reference(
            validation_acceptance["details"], validation_acceptance["valid"]
        )
        self.n_features_in_ = int(self.feature_dimensions_[-1])
        self.safe_mode_ = False
        self.safe_mode_reason_ = None
        self.gated_acceptance_refit_required_ = False
        _fit_end = time.perf_counter()
        self.stage_timing_ = {'fit_preparation_seconds': _preparation_seconds,
                             'predictor_training_calibration_seconds': _predictor_end - _predictor_begin,
                             'reliability_gate_seconds': _fit_end - _predictor_end,
                             'core_fit_total_seconds': _fit_end - _fit_begin}
        return self

    def _gated_acceptance_one(self, feature_vector: np.ndarray, info: Mapping[str, float], m: int) -> Tuple[bool, Optional[float]]:
        if (bool(getattr(self, "gated_acceptance_refit_required_", False))
                or not bool(self.gated_acceptance_schedule_.enabled_by_tau[m])):
            return False, None
        if self.selected_gated_acceptance_family_ == "conjunctive":
            for feature in self.gated_acceptance_features:
                threshold = float(self.gated_acceptance_schedule_.thresholds_by_feature[feature][m])
                value = float(info[feature])
                if self.FEATURE_DIRECTIONS[feature] == "high" and value < threshold:
                    return False, None
                if self.FEATURE_DIRECTIONS[feature] == "low" and value > threshold:
                    return False, None
            return True, None
        model = self.gated_acceptance_models_[m]
        if model is None:
            return False, None
        if (self.learned_history_mode == "eligible"
                and "action_stability" in self.gated_acceptance_features
                and int(info.get("available_prefix_count", m + 1)) < int(self.action_stability_window)):
            return False, None
        active_idx = [self.theory_feature_names_.index(name) for name in self.gated_acceptance_features]
        score = float(self._positive_probability(model, feature_vector[active_idx][None, :])[0])
        return score >= float(self.gated_acceptance_schedule_.learned_threshold_by_tau[m]), score

    def _predict_internal(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        check_is_fitted(self, attributes=["calibrators_", "gated_acceptance_schedule_", "classes_"])
        batch = self._validate_prediction_input(X)
        effective_times, fractions, terminal, valid = self._decision_geometry(batch)
        self.cost_model_.validate_time_cost_grid(np.unique(fractions[valid]))
        probabilities = self._probability_grid(batch)
        core_features, _, core_details = self._theory_feature_grid(
            probabilities, fractions, valid
        )
        reliability = self._predict_reliability(core_features, core_details, valid)
        theory_features, actions, details = self._theory_feature_grid(
            probabilities, fractions, valid, reliability
        )
        self._attach_reliability(details, reliability)
        n = batch.shape[0]
        predicted = np.zeros(n, dtype=int)
        stopping_times = np.zeros(n, dtype=int)
        stopped_probabilities = np.zeros((n, self.n_classes_), dtype=float)
        traces: List[Dict[str, Any]] = []
        for i in range(n):
            history: List[Dict[str, Any]] = []
            chosen = int(terminal[i])
            stop_reason = "forced_terminal"
            for m in range(int(terminal[i]) + 1):
                info = details[i][m]
                at_terminal = m == int(terminal[i])
                score: Optional[float] = None
                accepted = False
                if at_terminal:
                    should_stop = True
                    reason = "safe_mode_terminal" if bool(self.safe_mode_) else "forced_terminal"
                elif bool(self.safe_mode_):
                    should_stop = False
                    reason = "safe_mode_wait"
                else:
                    accepted, score = self._gated_acceptance_one(theory_features[i, m], info, m)
                    should_stop = bool(accepted)
                    reason = f"{self.selected_gated_acceptance_family_}_gated_acceptance" if accepted else "wait"
                raw_threshold = (
                    self.gated_acceptance_schedule_.threshold(int(self.tau_grid_[m]))
                    if not at_terminal else None
                )
                threshold = (
                    float(raw_threshold)
                    if raw_threshold is not None and np.isfinite(raw_threshold) else None
                )
                threshold_semantics = (
                    "not_applicable" if at_terminal else
                    "no_principal_pmax_threshold"
                    if self.selected_gated_acceptance_family_ == "conjunctive"
                    and "pmax" not in self.gated_acceptance_features else
                    "disabled" if threshold is None else "finite"
                )
                history.append(
                    {
                        "tau": int(self.tau_grid_[m]),
                        "effective_tau": int(effective_times[i, m]),
                        "tau_fraction": float(fractions[i, m]),
                        "pmax": float(info["pmax"]),
                        "margin": float(info["margin"]),
                        "entropy": float(info["entropy"]),
                        "action_stability": float(info["action_stability"]),
                        "risk_min": float(info["risk_min"]),
                        "risk_margin": float(info["risk_margin"]),
                        "reliability_score": float(info.get("reliability_score", reliability[i, m])),
                        "reliability": float(info.get("reliability_score", reliability[i, m])),
                        "gated_acceptance_family": self.selected_gated_acceptance_family_,
                        "variant": self.variant_,
                        "gated_acceptance_score": score,
                        "gated_acceptance_threshold": threshold,
                        "gated_acceptance_threshold_semantics": threshold_semantics,
                        "gated_acceptance_enabled": bool(
                            not at_terminal and not self.safe_mode_
                            and self.gated_acceptance_schedule_.enabled_by_tau[m]
                            and not getattr(self, "gated_acceptance_refit_required_", False)
                        ),
                        "gated_acceptance_accepted": bool(accepted),
                        "expected_total_cost": float(info["expected_total_cost"]),
                        "action_index": int(actions[i, m]),
                        "stop": bool(should_stop),
                        "reason": reason,
                    }
                )
                if should_stop:
                    chosen, stop_reason = m, reason
                    break
            predicted[i] = int(actions[i, chosen])
            stopping_times[i] = int(effective_times[i, chosen])
            stopped_probabilities[i] = probabilities[i, chosen]
            label = self.classes_[predicted[i]]
            traces.append(
                {
                    "prediction": label.item() if isinstance(label, np.generic) else label,
                    "prediction_index": int(predicted[i]),
                    "tau": int(stopping_times[i]),
                    "tau_fraction": float(fractions[i, chosen]),
                    "stop_reason": stop_reason,
                    "gated_acceptance_family": self.selected_gated_acceptance_family_,
                    "variant": self.variant_,
                    "safe_mode": bool(self.safe_mode_),
                    "history": history,
                }
            )
        labels = self.label_encoder_.inverse_transform(predicted)
        return labels, stopping_times, stopped_probabilities, traces

    def predict_proba_at_tau(self, X: ArrayLike, tau: Union[int, float]) -> np.ndarray:
        check_is_fitted(self, attributes=["calibrators_", "tau_grid_"])
        batch = self._validate_prediction_input(X)
        value = _finite_scalar(tau, "tau")
        if self.tau_input_mode_ == "fractional" and 0.0 < value <= 1.0:
            resolved = int(np.ceil(value * self.T_))
        else:
            resolved = _integer_control(value, "Absolute tau", 1)
        match = np.flatnonzero(self.tau_grid_ == resolved)
        if len(match) != 1:
            raise ValueError(
                f"tau={tau!r} resolves to {resolved}, not in {self.tau_grid_.tolist()}."
            )
        return self._probability_grid(batch)[:, int(match[0]), :]

    def _encode_fitted_labels(self, y: Sequence[Any], *, name: str = "y") -> np.ndarray:
        """Map validated labels exactly, without LabelEncoder dtype coercion."""
        check_is_fitted(self, attributes=["classes_"])
        values = _validate_labels(y, name)
        # Validate the combined domain before dictionary lookup: Python treats
        # booleans as integers, whereas the public label contract keeps them
        # separate and also distinguishes numeric labels from their strings.
        _validate_labels(list(self.classes_) + list(values), name)
        mapping = {value: index for index, value in enumerate(self.classes_)}
        try:
            return np.asarray([mapping[value] for value in values], dtype=int)
        except KeyError as exc:
            raise ValueError(f"{name} contains an unseen fitted class.") from exc

    def recalibrate(self, X: ArrayLike, y: Sequence[Any]) -> "GRACE":
        """Refit calibrators and force terminal-only safe mode until the gated-acceptance rule is refit."""
        check_is_fitted(self, attributes=["base_models_", "calibrators_", "gated_acceptance_schedule_"])
        batch = self._validate_prediction_input(X)
        y_encoded = self._encode_fitted_labels(y, name="Recalibration labels")
        if len(y_encoded) != batch.shape[0]:
            raise ValueError("y must align with X.")
        if not self._split_has_all_classes(y_encoded):
            raise ValueError("Recalibration data must contain every fitted class.")
        transformed = (
            self._apply_preprocessor(batch, self.final_preprocessor_)
            if self._uses_fold_preprocessing()
            else batch
        )
        updated = []
        for tau, model in zip(self.tau_grid_, self.base_models_):
            updated.append(
                self._fit_prefit_calibrator(
                    model, self._transform(transformed, int(tau)), y_encoded
                )
            )
        self.calibrators_ = updated
        self.gated_acceptance_schedule_.enabled_by_tau[:] = False
        if len(self.gated_acceptance_schedule_.enabled_by_tau):
            self.gated_acceptance_schedule_.enabled_by_tau[-1] = True
        self.gated_acceptance_refit_required_ = True
        self.safe_mode_ = True
        self.safe_mode_reason_ = "recalibration_requires_gate_refit"
        return self

    def export_gated_acceptance_schedule(self, path: Union[str, Path]) -> None:
        check_is_fitted(self, attributes=["gated_acceptance_schedule_"])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "grace-independent",
            "variant": self.variant_,
            "learned_history_mode": self.learned_history_mode,
            "action_space_scope": "A_equals_Y",
            "classes": [_json_ready(value) for value in self.classes_],
            "exact_risk_tie_rule": "first_class_in_fitted_classes_order",
            "series_length": int(self.T_),
            "channel_names": list(self.channel_names_),
            "fit_mode": self.fit_mode,
            "cost_matrix": self.cost_matrix_.tolist(),
            "grace_gated_acceptance": self.gated_acceptance_schedule_.as_dict(),
        }
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")



def multiclass_brier_score(y_encoded: np.ndarray, probabilities: np.ndarray) -> float:
    y_encoded, probabilities = _encoded_metric_inputs(y_encoded, probabilities)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_encoded]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_label_ece(
    y_encoded: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    if isinstance(n_bins, bool) or not np.isfinite(n_bins) or int(n_bins) != n_bins or n_bins < 2:
        raise ValueError("n_bins must be an integer at least two.")
    y_encoded, probabilities = _encoded_metric_inputs(y_encoded, probabilities)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (predicted == y_encoded).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if index == int(n_bins) - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return float(ece)


def evaluate_calibration_by_prefix(
    model: GRACE, X: ArrayLike, y: Sequence[Any], n_bins: int = 10
) -> List[Dict[str, Any]]:
    y_encoded = model._encode_fitted_labels(y, name="Evaluation labels")
    batch = model._validate_prediction_input(X)
    if len(y_encoded) != batch.shape[0]:
        raise ValueError("Evaluation labels must align with the prediction batch.")
    _, fractions, _, valid = model._decision_geometry(batch)
    probability_grid = model._probability_grid(batch)
    results: List[Dict[str, Any]] = []
    for m, tau in enumerate(model.tau_grid_):
        rows = np.flatnonzero(valid[:, m])
        if len(rows) == 0:
            results.append({"tau": float(tau), "n_evaluable": 0, "status": "not_evaluable",
                **{name: None for name in ("tau_fraction", "tau_fraction_mean", "tau_fraction_min", "tau_fraction_max",
                    "argmax_accuracy", "cost_aware_action_accuracy", "brier_score", "log_loss", "top_label_ece")}})
            continue
        probabilities = probability_grid[rows, m]
        y_at_tau = y_encoded[rows]
        probability_predictions = np.argmax(probabilities, axis=1)
        cost_actions = np.asarray(
            [
                model.cost_model_.bayes_action(
                    probability, tau_fraction=float(fractions[row, m])
                )
                for row, probability in zip(rows, probabilities)
            ],
            dtype=int,
        )
        row_fractions = np.asarray(fractions[rows, m], dtype=float)
        results.append(
            {
                "tau": float(tau),
                # For ragged series, the effective decision fraction is unit-specific.
                # ``tau_fraction`` is retained as the mean for compatibility.
                "tau_fraction": float(np.mean(row_fractions)),
                "tau_fraction_mean": float(np.mean(row_fractions)),
                "tau_fraction_min": float(np.min(row_fractions)),
                "tau_fraction_max": float(np.max(row_fractions)),
                "n_evaluable": int(len(rows)),
                "status": "ok",
                "argmax_accuracy": float(np.mean(probability_predictions == y_at_tau)),
                "cost_aware_action_accuracy": float(np.mean(cost_actions == y_at_tau)),
                "brier_score": multiclass_brier_score(y_at_tau, probabilities),
                "log_loss": float(
                    log_loss(y_at_tau, _clip_probabilities_for_logloss(probabilities), labels=np.arange(model.n_classes_))
                ),
                "top_label_ece": top_label_ece(y_at_tau, probabilities, n_bins=n_bins),
            }
        )
    return results


def _discrimination_metrics(y_encoded: np.ndarray, probabilities: np.ndarray) -> Dict[str, Optional[float]]:
    """Binary class-1 AUROC/AP or fixed-K macro one-vs-rest scores.

    A draw missing any configured class returns None for both scores. This
    preserves the estimand across point estimates and bootstrap replicates.
    """
    y_encoded, probabilities = _encoded_metric_inputs(y_encoded, probabilities)
    observed = np.unique(y_encoded)
    if len(observed) != probabilities.shape[1]:
        return {"auroc_at_stop": None, "auprc_at_stop": None}

    auc_values: List[float] = []
    ap_values: List[float] = []
    for class_index in ([1] if probabilities.shape[1] == 2 else observed):
        class_index = int(class_index)
        if class_index < 0 or class_index >= probabilities.shape[1]:
            continue
        binary = (y_encoded == class_index).astype(int)
        positives = int(binary.sum())
        negatives = int(len(binary) - positives)
        if positives == 0 or negatives == 0:
            continue
        score = probabilities[:, class_index]
        try:
            auc_values.append(float(roc_auc_score(binary, score)))
            ap_values.append(float(average_precision_score(binary, score)))
        except ValueError:
            continue
    if not auc_values or not ap_values:
        return {"auroc_at_stop": None, "auprc_at_stop": None}
    return {
        "auroc_at_stop": float(np.mean(auc_values)),
        "auprc_at_stop": float(np.mean(ap_values)),
    }


def evaluate_grace(
    model: GRACE, X: ArrayLike, y: Sequence[Any], n_bins: int = 10
) -> Dict[str, Any]:
    y_encoded = model._encode_fitted_labels(y, name="Evaluation labels")
    batch = model._validate_prediction_input(X)
    if len(y_encoded) != len(batch):
        raise ValueError("Evaluation labels must align with the prediction batch.")
    y_array = model.classes_[y_encoded]
    labels, times, probabilities, traces = model.predict_early_proba(batch)
    predicted_encoded = model.label_encoder_.transform(labels).astype(int)
    decision_fractions = np.asarray(
        [float(trace["tau_fraction"]) for trace in traces], dtype=float
    )

    actual_misclassification = np.asarray(
        [
            model.cost_model_.actual_misclassification_cost(yt, yp)
            for yt, yp in zip(y_encoded, predicted_encoded)
        ],
        dtype=float,
    )
    delay_components = np.asarray(
        [
            model.cost_model_.delay_component(tau_fraction=float(fraction))
            for fraction in decision_fractions
        ],
        dtype=float,
    )
    delay_penalties = np.asarray(
        [
            model.cost_model_.delay_penalty(tau_fraction=float(fraction))
            for fraction in decision_fractions
        ],
        dtype=float,
    )
    observation_penalties = np.asarray(
        [
            model.cost_model_.observation_penalty(tau_fraction=float(fraction))
            for fraction in decision_fractions
        ],
        dtype=float,
    )
    total_cost = np.asarray(
        [
            model.cost_model_.total_cost(
                int(yt), int(yp), tau_fraction=float(fraction)
            )
            for yt, yp, fraction in zip(
                y_encoded, predicted_encoded, decision_fractions
            )
        ],
        dtype=float,
    )
    reasons = [str(trace["stop_reason"]) for trace in traces]
    stop_distribution = {
        str(int(tau)): int(np.sum(times == tau)) for tau in np.unique(times)
    }
    discrimination = _discrimination_metrics(y_encoded, probabilities)

    return {
        "n_test": int(len(y_array)),
        "classes": [_json_ready(value) for value in model.classes_],
        "accuracy": float(accuracy_score(y_array, labels)),
        "macro_f1": float(
            f1_score(y_array, labels, labels=model.classes_, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(np.mean([np.mean(labels[y_array==c]==c) for c in np.unique(y_array)])),
        "avg_tau": float(np.mean(times)),
        "decision_fraction": float(np.mean(decision_fractions)),
        "earliness_gain": float(1.0 - np.mean(decision_fractions)),
        "avg_misclassification_cost": float(np.mean(actual_misclassification)),
        "avg_delay_component": float(np.mean(delay_components)),
        "avg_delay_penalty": float(np.mean(delay_penalties)),
        "avg_observation_cost": float(np.mean(observation_penalties)),
        "avg_total_cost": float(np.mean(total_cost)),
        "brier_score_at_stop": multiclass_brier_score(y_encoded, probabilities),
        "log_loss_at_stop": float(
            log_loss(y_encoded, _clip_probabilities_for_logloss(probabilities), labels=np.arange(model.n_classes_))
        ),
        "top_label_ece_at_stop": top_label_ece(
            y_encoded, probabilities, n_bins=n_bins
        ),
        **discrimination,
        "conjunctive_gated_acceptance_rate": float(np.mean(np.asarray(reasons) == "conjunctive_gated_acceptance")),
        "learned_gated_acceptance_rate": float(np.mean(np.asarray(reasons) == "learned_gated_acceptance")),
        "gated_acceptance_stop_rate": float(np.mean(np.isin(np.asarray(reasons), ["conjunctive_gated_acceptance", "learned_gated_acceptance"]))),
        "forced_terminal_rate": float(
            np.mean(np.asarray(reasons) == "forced_terminal")
        ),
        "safe_mode_terminal_rate": float(
            np.mean(np.asarray(reasons) == "safe_mode_terminal")
        ),
        "stop_distribution": stop_distribution,
        "confusion_matrix": confusion_matrix(
            y_array, labels, labels=model.classes_
        ).astype(int).tolist(),
        "grace_calibration_by_prefix": evaluate_calibration_by_prefix(
            model, X, y, n_bins=n_bins
        ),
        "grace_gated_acceptance_validation": model.gated_acceptance_schedule_.validation_stats,
    }


# ============================================================================
# Synthetic demonstration and CLI
# ============================================================================


def generate_synthetic_early_dataset(
    n_samples: int = 900,
    T: int = 60,
    n_classes: int = 3,
    n_channels: int = 2,
    noise: float = 0.65,
    random_state: int = 42,
    label_names: Optional[Sequence[Any]] = ("SELL", "HOLD", "BUY"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a non-clinical proof-of-concept ETSC dataset."""
    n_samples = _integer_control(n_samples, "n_samples", 1)
    T = _integer_control(T, "T", 1)
    n_classes = _integer_control(n_classes, "n_classes", 2)
    n_channels = _integer_control(n_channels, "n_channels", 1)
    noise = _finite_scalar(noise, "noise")
    if noise < 0.0:
        raise ValueError("noise must be non-negative.")
    if n_classes < 2 or n_samples < n_classes * 8:
        raise ValueError("Use at least two classes and at least eight samples per class.")
    if T < 8 or n_channels < 1:
        raise ValueError("T must be >=8 and n_channels must be positive.")
    if label_names is not None:
        names = _validate_labels(label_names, "label_names")
        if len(names) != n_classes or len(set(names.tolist())) != n_classes:
            raise ValueError("label_names must have exactly one distinct valid name per class.")

    rng = np.random.default_rng(random_state)
    class_indices = np.arange(n_samples, dtype=int) % n_classes
    rng.shuffle(class_indices)
    time = np.linspace(0.0, 1.0, T)
    X = np.zeros((n_samples, T, n_channels), dtype=float)

    centered = np.linspace(-1.0, 1.0, n_classes)
    for i, cls in enumerate(class_indices):
        direction = centered[cls]
        phase = (cls + 1) * np.pi / (n_classes + 1)
        subject_intercept = rng.normal(0.0, 0.20)
        subject_slope = rng.normal(0.0, 0.16)
        cue = direction * (0.30 * np.minimum(time / 0.30, 1.0))
        latent = (
            0.70 * direction * time
            + 0.28 * np.sin(2.0 * np.pi * time + phase)
            + cue
            + subject_intercept
            + subject_slope * time
        )
        X[i, :, 0] = latent + rng.normal(0.0, noise, T)
        for channel in range(1, n_channels):
            scale = 0.20 + 0.08 * abs(direction) + 0.03 * channel
            X[i, :, channel] = (
                scale * np.cos((channel + 1) * np.pi * time + phase)
                + 0.18 * direction
                + rng.normal(0.0, noise * 0.75, T)
            )

    if label_names is None:
        labels = class_indices
    else:
        names = np.asarray(label_names, dtype=object)
        labels = names[class_indices]
    return X, np.asarray(labels)


def run_demo(
    random_state: int = 42,
    split_strategy: str = "stratified",
    n_samples: int = 900,
    quick: bool = False,
) -> Tuple[GRACE, Dict[str, Any]]:
    if split_strategy not in {"stratified", "chronological"}:
        raise ValueError("The demo supports only stratified or chronological splitting.")
    n_samples = _integer_control(n_samples, "n_samples", 1)
    if quick:
        n_samples = min(int(n_samples), 450)
    X, y = generate_synthetic_early_dataset(
        n_samples=int(n_samples), random_state=int(random_state)
    )

    if split_strategy == "chronological":
        cut = int(round(0.75 * len(y)))
        X_train, X_test = X[:cut], X[cut:]
        y_train, y_test = y[:cut], y[cut:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=int(random_state),
        )

    base = RandomForestClassifier(
        n_estimators=60 if quick else 140,
        min_samples_leaf=2,
        random_state=int(random_state),
        n_jobs=-1,
    )
    cost_matrix = np.asarray(
        [
            [0.0, 1.0, 2.5],
            [1.2, 0.0, 1.2],
            [2.5, 1.0, 0.0],
        ]
    )
    model = GRACE(
        prefix_model=base,
        decision_points=(0.20, 0.35, 0.50, 0.65, 0.80, 1.00),
        calibration_method="sigmoid",
        split_strategy=split_strategy,
        gated_acceptance_family="conjunctive",
        action_stability_window=2,
        alpha_delay=0.35,
        misclassification_cost=cost_matrix,
        cost_class_order=("SELL", "HOLD", "BUY"),
        random_state=int(random_state),
    )
    if split_strategy == "chronological":
        # Independent synthetic episodes have no overlapping information
        # intervals. This artificial clock demonstrates the temporal API; it
        # does not claim that the generator models a financial time series.
        clock = pd.date_range("2000-01-01", periods=len(y_train), freq="D", tz="UTC")
        model.fit(X_train, y_train, episode_time=clock, information_start=clock, information_end=clock)
    else:
        model.fit(X_train, y_train)
    metrics = evaluate_grace(model, X_test, y_test)
    metrics["internal_split_summary"] = model.split_summary_
    metrics["data_scope"] = "synthetic_execution_example_not_empirical_financial_evidence"
    metrics["grace_gated_acceptance_validation"] = model.gated_acceptance_schedule_.validation_stats
    return model, metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the independent GRACE synthetic demonstration."
    )
    parser.add_argument("--demo", action="store_true", help="Run the synthetic demo.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller, faster demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-samples", type=int, default=900, help="Synthetic sample count.")
    parser.add_argument(
        "--split-strategy",
        choices=("stratified", "chronological"),
        default="stratified",
        help="Internal training split strategy.",
    )
    parser.add_argument(
        "--export-gated-acceptance", type=str, default="", help="Optional schedule JSON path."
    )
    parser.add_argument(
        "--export-metrics", type=str, default="", help="Optional metrics JSON path."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.demo:
        print(
            "GRACE loaded. Use --demo for a synthetic run or import GRACE "
            "from grace_core.py."
        )
        return
    model, metrics = run_demo(
        random_state=args.seed,
        split_strategy=args.split_strategy,
        n_samples=args.n_samples,
        quick=args.quick,
    )
    if args.export_gated_acceptance:
        model.export_gated_acceptance_schedule(args.export_gated_acceptance)
    if args.export_metrics:
        _write_json(args.export_metrics, metrics)
    print(json.dumps(_json_ready(metrics), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
