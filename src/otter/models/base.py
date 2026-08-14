"""FGWModel, base class for all cross-species coupling models.

The four production subclasses (UnsupervisedGW, SupervisedFGW, MultimodalFGW,
HierarchicalFGW) all inherit from this. Their :meth:`fit` solves the underlying
FGW problem and stores the resulting coupling π on ``self.pi_``.

The class bundles configuration (alpha, epsilon, weights, …) with the
resulting π and the diagnostic info (loss, n_iter, multistart spread), and
gives a uniform ``predict_human_fc`` and ``evaluate`` interface across the
four model levels.

Subclass contract
-----------------
A subclass must implement:
    _solve(self, *, mouse_ad, human_ad, **kw) -> (pi, info_dict)

The base class then handles ``.fit``, ``.pi``, ``.predict_human_fc``,
``.evaluate``, ``.save`` and ``.load`` for free.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class FitInfo:
    """Solver-side diagnostics returned by every FGWModel.fit()."""
    loss: float = float("nan")
    n_iter: int = 0
    converged: bool = True
    n_restarts: int = 1
    extra: dict = field(default_factory=dict)


class FGWModel:
    """Base class for cross-species FGW models.

    Subclasses set their own __init__ defaults and implement :meth:`_solve`.
    :meth:`fit` calls _solve and stores the results; it is not overridden.
    """

    #: Human-readable name for reporting. Subclasses override.
    _name: str = "FGWModel"

    def __init__(self, **config):
        self.config = dict(config)
        self.pi_: Optional[np.ndarray] = None
        self.fit_info_: Optional[FitInfo] = None
        self._mouse_ad = None
        self._human_ad = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, mouse_ad, human_ad, **fit_kwargs) -> "FGWModel":
        """Solve the FGW problem. Stores ``self.pi_`` and ``self.fit_info_``."""
        self._mouse_ad = mouse_ad
        self._human_ad = human_ad
        pi, info = self._solve(mouse_ad=mouse_ad, human_ad=human_ad, **fit_kwargs)
        if pi is None or pi.ndim != 2:
            raise RuntimeError(f"{type(self).__name__}._solve returned invalid pi")
        self.pi_ = np.asarray(pi, dtype=np.float64)
        self.fit_info_ = info if isinstance(info, FitInfo) else FitInfo(**(info or {}))
        return self

    @property
    def pi(self) -> np.ndarray:
        """The (n_m, n_h) cross-species coupling. Raises if not yet fit."""
        if self.pi_ is None:
            raise RuntimeError("Model not fit yet, call .fit(mouse_ad, human_ad) first.")
        return self.pi_

    def predict_human_fc(self, mouse_fc: Optional[np.ndarray] = None,
                          *, eps: float = 1e-12) -> np.ndarray:
        """Push mouse FC through π to predict human FC.

        Default mouse_fc = the mean FC stored on the AnnData passed at fit time.
        """
        from otter.eval.translation import predict_human_fc
        if mouse_fc is None:
            if self._mouse_ad is None:
                raise ValueError("no mouse_fc and no fit() to fall back to")
            mouse_fc = self._mouse_ad.uns["fc_mean"]
        return predict_human_fc(self.pi, mouse_fc, eps=eps)

    def evaluate(
        self, *,
        held_out_pair_ids: Optional[list[int]] = None,
        eval_kind: str = "anchor",
    ) -> dict:
        """One-shot evaluation. eval_kind ∈ {'anchor', 'translation'}.

        For ``anchor``: returns held-out anchor recovery metrics. If
        held_out_pair_ids is None, evaluates on ALL anchors (full supervision).
        For ``translation``: FC-translation Pearson r vs the true human FC.
        """
        if eval_kind == "anchor":
            from otter.data.anchors import (
                get_anchor_index, held_out_metrics_graded, metrics_summary,
            )
            idx_m = get_anchor_index(self._mouse_ad.var)
            idx_h = get_anchor_index(self._human_ad.var)
            pi_anchor = self.pi[np.ix_(idx_m.pos, idx_h.pos)]
            if held_out_pair_ids is None:
                return metrics_summary(pi_anchor, idx_m, idx_h)
            return held_out_metrics_graded(
                pi_anchor, idx_m, idx_h, held_out_pair_ids,
                var_h=self._human_ad.var,
            )
        if eval_kind == "translation":
            from otter.data.networks import assign_networks
            from otter.data.anchors import get_anchor_index
            from otter.eval.translation import fc_translation_quality
            idx_h = get_anchor_index(self._human_ad.var)
            net_h = assign_networks(self._human_ad.var, idx_h)
            return fc_translation_quality(
                self.pi.astype(np.float64),
                self._mouse_ad.uns["fc_mean"].astype(np.float64),
                self._human_ad.uns["fc_mean"].astype(np.float64),
                network_labels_h=net_h,
            )
        raise ValueError(f"unknown eval_kind: {eval_kind}")

    def save(self, path: str | Path, *, save_info: bool = True) -> None:
        """Save π as a .npy. If save_info, also writes a sidecar JSON with
        config + fit diagnostics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.pi.astype(np.float32))
        if save_info:
            sidecar = path.with_suffix(".json")
            sidecar.write_text(json.dumps({
                "model_class": type(self).__name__,
                "config":      self.config,
                "fit_info":    asdict(self.fit_info_) if self.fit_info_ else None,
                "shape":       list(self.pi.shape),
                "pi_file":     path.name,
            }, indent=2, default=float))

    @classmethod
    def load(cls, path: str | Path) -> "FGWModel":
        """Load a saved π file (and optional sidecar) into a model instance.

        The loaded model is reconstituted: it has self.pi_ but no adata refs,
        so .predict_human_fc and .evaluate require explicit arguments
        (mouse_fc, mouse_ad, human_ad).
        """
        path = Path(path)
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            cls_name = meta.get("model_class", cls.__name__)
            # Map the model class name back to the class object
            cls_map = {c.__name__: c for c in _all_subclasses(FGWModel)}
            target_cls = cls_map.get(cls_name, cls)
            inst = target_cls(**meta.get("config", {}))
            if meta.get("fit_info"):
                inst.fit_info_ = FitInfo(**meta["fit_info"])
        else:
            inst = cls()
        inst.pi_ = np.load(path).astype(np.float64)
        return inst

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    def _solve(self, *, mouse_ad, human_ad, **kw) -> tuple[np.ndarray, FitInfo | dict]:
        """Subclass-specific solve. Must return (pi, fit_info_or_dict)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        if self.pi_ is None:
            return f"{type(self).__name__}(not fit)"
        loss = self.fit_info_.loss if self.fit_info_ else float("nan")
        return (f"{type(self).__name__}(pi.shape={self.pi.shape}, "
                f"loss={loss:.5f})")


def _all_subclasses(cls):
    """All transitive subclasses of cls, for save/load class lookup."""
    out = set(cls.__subclasses__())
    for sub in list(out):
        out |= _all_subclasses(sub)
    return out
