"""
factor_pool_manager.py

Manages the persistent universal factor pool.
Pool file: reports/factor_pool.json

Each entry is keyed by a unique param-combo string so different param sets
for the same factor name can coexist. Entries include the OOS equity curve
(needed for cross-factor correlation checks on future additions).
"""

import json
import logging
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_DEFAULT_POOL_PATH = Path(__file__).resolve().parent.parent / "reports" / "factor_pool.json"


class FactorPoolManager:
    """
    Persistent universal factor pool.

    Pool JSON structure:
    {
      "last_updated": "2026-03-27T12:00:00",
      "factors": {
        "<pool_key>": {
          "pool_key": str,
          "factor_id": str,
          "asset": str,
          "factor_name": str,
          "parameters": dict,
          "is_sharpe": float,
          "oos_sharpe": float,
          "oos_calmar": float,
          "oos_annual_return": float,
          "oos_max_drawdown": float,
          "oos_num_trades": int,
          "oos_equity_curve": list[float],
          "status": str,
          "added_date": str,
          "last_validated": str,
          "source_report_date": str
        }
      }
    }
    """

    def __init__(self, pool_path: Path = None, corr_threshold: float = 0.7):
        self.pool_path = Path(pool_path) if pool_path else _DEFAULT_POOL_PATH
        self.corr_threshold = corr_threshold
        self._pool = self._load()

    def _load(self) -> Dict:
        if self.pool_path.exists():
            try:
                with open(self.pool_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.error(f"Failed to load pool from {self.pool_path}: {e}")
        return {"last_updated": None, "factors": {}}

    def save(self):
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool["last_updated"] = datetime.now().isoformat()
        with open(self.pool_path, "w", encoding="utf-8") as f:
            json.dump(self._pool, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Pool saved: {len(self._pool['factors'])} factors -> {self.pool_path}")

    @staticmethod
    def make_pool_key(factor_id: str, params: Dict) -> str:
        """
        Build a unique string key for a factor+param combo.
        Format: <factor_id>__<method>_r<rolling>_l<long_param>_s<short_param>
        """
        method = params.get("param_method", "unknown")
        rolling = params.get("rolling", 0)
        lp = params.get("long_param", 0)
        sp = params.get("short_param", 0)
        return f"{factor_id}__{method}_r{rolling}_l{lp}_s{sp}"

    def get_all_factors(self) -> Dict:
        """Return a copy of all pool entries."""
        return dict(self._pool["factors"])

    def get_pool_summary(self) -> Dict:
        factors = self._pool["factors"]
        if not factors:
            return {"total": 0, "by_asset": {}, "avg_oos_sharpe": 0.0}
        sharpes = [v.get("oos_sharpe", 0.0) for v in factors.values() if v.get("oos_sharpe")]
        by_asset = dict(Counter(v.get("asset", "unknown") for v in factors.values()))
        return {
            "total": len(factors),
            "by_asset": by_asset,
            "avg_oos_sharpe": round(float(np.mean(sharpes)), 4) if sharpes else 0.0,
        }

    def _compute_corr(self, a: list, b: list) -> float:
        """Pearson correlation between two equity curves, aligned to min length."""
        min_len = min(len(a), len(b))
        if min_len < 2:
            return 0.0
        arr_a = np.array(a[:min_len], dtype=float)
        arr_b = np.array(b[:min_len], dtype=float)
        if arr_a.std() == 0 or arr_b.std() == 0:
            return 0.0
        return float(np.corrcoef(arr_a, arr_b)[0, 1])

    def _make_entry(self, factor_record: Dict, status: str, source_date: str = None) -> Dict:
        """Build a pool entry dict from a Stage-1 result record."""
        return {
            "pool_key": self.make_pool_key(
                factor_record["factor_id"], factor_record.get("parameters", {})
            ),
            "factor_id": factor_record["factor_id"],
            "asset": factor_record.get("asset", ""),
            "factor_name": factor_record.get("factor_name", ""),
            "parameters": factor_record.get("parameters", {}),
            "is_sharpe": factor_record.get("is_sharpe", 0.0),
            "oos_sharpe": factor_record.get("oos_sharpe", 0.0),
            "oos_calmar": factor_record.get("oos_calmar", 0.0),
            "oos_annual_return": factor_record.get("oos_annual_return", 0.0),
            "oos_max_drawdown": factor_record.get("oos_max_drawdown", 0.0),
            "oos_num_trades": factor_record.get("oos_num_trades", 0),
            "oos_equity_curve": factor_record.get("oos_equity_curve", []),
            "status": status,
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "last_validated": datetime.now().strftime("%Y-%m-%d"),
            "source_report_date": source_date or datetime.now().strftime("%Y-%m-%d"),
        }

    def add_factor(self, factor_record: Dict, source_date: str = None) -> tuple:
        """
        Try to add a validated factor to the pool.

        Decision logic:
        1. If same pool_key exists: keep whichever has higher OOS Sharpe.
        2. Else: add unconditionally (correlation is not checked).

        Returns:
            (action, reason) where action is one of:
            "added" | "replaced" | "rejected_weaker"
        """
        key = self.make_pool_key(
            factor_record["factor_id"], factor_record.get("parameters", {})
        )
        new_sharpe = factor_record.get("oos_sharpe", 0.0)
        new_curve = factor_record.get("oos_equity_curve", [])

        # Same-key update
        existing = self._pool["factors"].get(key)
        if existing:
            if new_sharpe > existing.get("oos_sharpe", 0.0):
                self._pool["factors"][key] = self._make_entry(
                    factor_record, "in_pool", source_date
                )
                logger.info(
                    f"  REPLACED [{key}] sharpe {existing.get('oos_sharpe', 0):.3f} -> {new_sharpe:.3f}"
                )
                return "replaced", "same_key_better_sharpe"
            logger.info(
                f"  SKIP [{key}] same key, weaker sharpe "
                f"({new_sharpe:.3f} <= {existing.get('oos_sharpe', 0):.3f})"
            )
            return "rejected_weaker", "same_key_lower_sharpe"

        # No conflict: add
        self._pool["factors"][key] = self._make_entry(
            factor_record, "in_pool", source_date
        )
        logger.info(f"  ADDED [{key}] oos_sharpe={new_sharpe:.3f}")
        return "added", "no_correlation_conflict"
