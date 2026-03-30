"""
factor_validation_agent.py

Validates promising factors from the daily backtest report using two stages:
  Stage 1 - Walk-forward out-of-sample (OOS) test: re-run each factor on the
            unseen 35% of the date range using the same optimized parameters.
  Stage 2 - Correlation deduplication: cluster factors whose OOS equity curves
            are highly correlated and keep only the best from each cluster.

Output: reports/validation_reports/validation_YYYY-MM-DD.json
"""

import sys
import os
import json
import logging
import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
from config.trading_config import DEFAULT_INITIAL_CAPITAL
from core.factor_pool_manager import FactorPoolManager

# Logging setup
_project_root = Path(__file__).resolve().parent.parent
_logs_dir = _project_root / "logs"
_logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            _logs_dir / f"factor_validation_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TRAIN_PCT = 0.65
DEFAULT_OOS_SHARPE_THRESHOLD = 0.8
DEFAULT_SHARPE_DECAY_THRESHOLD = 0.4
DEFAULT_CORRELATION_THRESHOLD = 0.7
DEFAULT_MIN_OOS_TRADES = 5
DEFAULT_MIN_OOS_DAYS = 90


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _find_latest_report(reports_dir: Path) -> Path:
    """Return path to the most recent daily report JSON."""
    candidates = sorted(reports_dir.glob("report_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No report_*.json found in {reports_dir}")
    return candidates[-1]


def _split_date_range(
    start: str, end: str, train_pct: float = DEFAULT_TRAIN_PCT
) -> Tuple[str, str]:
    """
    Return (oos_start, oos_end) strings given full date range and train fraction.

    The OOS window starts the day after the train period ends.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    train_days = int(total_days * train_pct)
    train_end_dt = start_dt + timedelta(days=train_days)
    oos_start_dt = train_end_dt + timedelta(days=1)
    return oos_start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _compute_calmar(annual_return: float, max_drawdown: float) -> float:
    """Calmar ratio: annual_return / max_drawdown (both as decimals)."""
    if max_drawdown <= 0:
        return 0.0
    return annual_return / max_drawdown


def _cluster_by_correlation(
    corr_matrix: pd.DataFrame, threshold: float
) -> List[List[int]]:
    """
    Union-find clustering: indices i and j go into the same cluster if
    their pairwise correlation exceeds threshold.

    Returns list of clusters, each cluster being a list of column indices.
    """
    n = len(corr_matrix)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix.iloc[i, j] > threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class FactorValidationAgent:
    """
    Two-stage factor validation pipeline.

    Stage 1: Walk-forward OOS test.
    Stage 2: Correlation deduplication.
    """

    def __init__(self, config: Dict):
        self.train_pct = config.get("train_pct", DEFAULT_TRAIN_PCT)
        self.oos_sharpe_threshold = config.get(
            "oos_sharpe_threshold", DEFAULT_OOS_SHARPE_THRESHOLD
        )
        self.sharpe_decay_threshold = config.get(
            "sharpe_decay_threshold", DEFAULT_SHARPE_DECAY_THRESHOLD
        )
        self.correlation_threshold = config.get(
            "correlation_threshold", DEFAULT_CORRELATION_THRESHOLD
        )
        self.min_oos_trades = config.get("min_oos_trades", DEFAULT_MIN_OOS_TRADES)
        self.min_oos_days = config.get("min_oos_days", DEFAULT_MIN_OOS_DAYS)
        self.report_path = config.get("report_path", None)
        self.last_n_days = config.get("last_n_days", None)
        self.process_all = config.get("process_all", False)

        self._project_root = Path(__file__).resolve().parent.parent
        self._reports_dir = self._project_root / "reports" / "daily_reports"
        self._output_dir = self._project_root / "reports" / "validation_reports"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.pool = FactorPoolManager(corr_threshold=self.correlation_threshold)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        logger.info("=== Factor Validation Agent started ===")

        factors = self._load_reports()
        logger.info(f"Loaded {len(factors)} factors for validation")

        stage1_results, stage1_passed = self._stage1_oos_validation(factors)
        logger.info(
            f"Stage 1 complete: {len(stage1_passed)}/{len(factors)} factors passed OOS test"
        )

        pool_actions = self._stage2_pool_update(stage1_passed)
        self.pool.save()
        pool_summary = self.pool.get_pool_summary()
        logger.info(
            f"Stage 2 complete: {pool_actions['added']} added, "
            f"{pool_actions['replaced']} replaced, "
            f"{pool_actions['rejected_corr']} rejected (corr), "
            f"{pool_actions['rejected_weaker']} rejected (weaker). "
            f"Pool total: {pool_summary['total']}"
        )

        report = self._build_report(factors, stage1_results, pool_actions, pool_summary)
        out_path = self._save_report(report)
        logger.info(f"Validation report saved to: {out_path}")
        logger.info("=== Factor Validation Agent finished ===")
        return report

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_reports(self) -> List[Dict]:
        """
        Load factors from one or more daily reports.
        Priority: --report-path > --last-n-days > --all > latest only.
        Returns flat list of factor dicts, each with 'source_report_date' added.
        """
        if self.report_path:
            return self._read_report_file(Path(self.report_path))

        candidates = sorted(self._reports_dir.glob("report_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No report_*.json found in {self._reports_dir}")

        if self.process_all:
            selected = candidates
        elif self.last_n_days:
            selected = candidates[-self.last_n_days:]
        else:
            selected = [candidates[-1]]

        all_factors = []
        for path in selected:
            all_factors.extend(self._read_report_file(path))
        logger.info(f"Loaded {len(all_factors)} factors from {len(selected)} report(s)")
        return all_factors

    def _read_report_file(self, path: Path) -> List[Dict]:
        logger.info(f"Reading report: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        date_str = path.stem.replace("report_", "")
        factors = data.get("promising_factors_ranked", [])
        if not factors:
            logger.warning(f"No promising_factors_ranked in {path}")
            return []
        for f in factors:
            f["source_report_date"] = date_str
        return factors

    # ------------------------------------------------------------------
    # Stage 1: OOS backtest
    # ------------------------------------------------------------------

    def _stage1_oos_validation(
        self, factors: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Re-run each factor on the OOS period using the same optimized parameters.

        Returns (all_results, passing_results).
        """
        all_results = []
        passing = []

        for factor in factors:
            factor_id = factor.get("factor_id", "unknown")
            result = self._run_single_oos(factor)
            all_results.append(result)

            if result["passed"]:
                passing.append(result)
            else:
                logger.info(
                    f"  FAIL [{factor_id}] reason={result['fail_reason']} "
                    f"oos_sharpe={result.get('oos_sharpe', 'n/a')}"
                )

        return all_results, passing

    def _run_single_oos(self, factor: Dict) -> Dict:
        """Run OOS backtest for one factor and return a result record."""
        factor_id = factor.get("factor_id", "unknown")
        asset = factor.get("asset", "BTC")
        factor_name = factor.get("factor_name", "")
        is_sharpe = factor.get("sharpe_ratio", 0.0)
        params = factor.get("parameters", {})

        base = {
            "factor_id": factor_id,
            "asset": asset,
            "factor_name": factor_name,
            "is_sharpe": is_sharpe,
            "oos_sharpe": None,
            "oos_calmar": None,
            "oos_annual_return": None,
            "oos_max_drawdown": None,
            "oos_num_trades": None,
            "sharpe_decay_ratio": None,
            "oos_equity_curve": [],
            "parameters": params,
            "passed": False,
            "fail_reason": None,
        }

        # Determine OOS date window
        date_range = params.get("date_range")
        if not date_range or len(date_range) != 2:
            base["fail_reason"] = "missing_date_range"
            logger.warning(f"  [{factor_id}] no date_range in parameters, skipping")
            return base

        oos_start, oos_end = _split_date_range(
            date_range[0], date_range[1], self.train_pct
        )

        # Check OOS window length
        oos_start_dt = datetime.strptime(oos_start, "%Y-%m-%d")
        oos_end_dt = datetime.strptime(oos_end, "%Y-%m-%d")
        oos_days = (oos_end_dt - oos_start_dt).days
        if oos_days < self.min_oos_days:
            base["fail_reason"] = "insufficient_oos_data"
            logger.warning(
                f"  [{factor_id}] OOS window only {oos_days} days, need {self.min_oos_days}"
            )
            return base

        # Build OOS params (same params, new date_range)
        oos_params = dict(params)
        oos_params["date_range"] = [oos_start, oos_end]

        # Instantiate fresh strategy (param_method baked into constructor)
        param_method = params.get("param_method", "pct_change")
        strategy = EnhancedNonPriceStrategy(
            initial_capital=params.get("initial_capital", DEFAULT_INITIAL_CAPITAL),
            min_lot_size=params.get("lot_size", 0.001),
            param_method=param_method,
        )

        logger.info(
            f"  Running OOS [{factor_id}] period={oos_start} to {oos_end} ..."
        )
        try:
            result = strategy.run_backtest(asset, factor_name, oos_params)
        except Exception as e:
            base["fail_reason"] = "backtest_exception"
            logger.error(f"  [{factor_id}] backtest exception: {e}")
            return base

        if result is None:
            base["fail_reason"] = "backtest_failed"
            logger.warning(f"  [{factor_id}] run_backtest returned None")
            return base

        oos_sharpe = result.get("sharpe_ratio", 0.0)
        oos_annual_return = result.get("annual_return", 0.0)
        oos_max_drawdown = result.get("max_drawdown", 0.0)
        oos_num_trades = result.get("num_trades", 0)
        oos_equity_curve = result.get("equity_curve", [])
        oos_calmar = _compute_calmar(oos_annual_return, oos_max_drawdown)

        base["oos_sharpe"] = round(oos_sharpe, 4)
        base["oos_calmar"] = round(oos_calmar, 4)
        base["oos_annual_return"] = round(oos_annual_return, 4)
        base["oos_max_drawdown"] = round(oos_max_drawdown, 4)
        base["oos_num_trades"] = oos_num_trades
        base["oos_equity_curve"] = oos_equity_curve

        # Minimum trades check
        if oos_num_trades < self.min_oos_trades:
            base["fail_reason"] = "insufficient_oos_trades"
            logger.info(
                f"  [{factor_id}] only {oos_num_trades} trades in OOS period"
            )
            return base

        # Decay ratio (clamp to 0 if OOS Sharpe is negative)
        if oos_sharpe <= 0 or is_sharpe <= 0:
            decay_ratio = 0.0
        else:
            decay_ratio = round(oos_sharpe / is_sharpe, 4)
        base["sharpe_decay_ratio"] = decay_ratio

        # Apply filters
        if oos_sharpe < self.oos_sharpe_threshold:
            base["fail_reason"] = "oos_sharpe_below_threshold"
            logger.info(
                f"  [{factor_id}] OOS Sharpe {oos_sharpe:.3f} < {self.oos_sharpe_threshold}"
            )
            return base

        if decay_ratio < self.sharpe_decay_threshold:
            base["fail_reason"] = "severe_sharpe_decay"
            logger.info(
                f"  [{factor_id}] decay ratio {decay_ratio:.3f} < {self.sharpe_decay_threshold} "
                f"(IS={is_sharpe:.2f} OOS={oos_sharpe:.2f})"
            )
            return base

        base["passed"] = True
        logger.info(
            f"  PASS [{factor_id}] OOS Sharpe={oos_sharpe:.3f} "
            f"decay={decay_ratio:.3f} calmar={oos_calmar:.3f} trades={oos_num_trades}"
        )
        return base

    # ------------------------------------------------------------------
    # Stage 2: Correlation deduplication
    # ------------------------------------------------------------------

    def _stage2_pool_update(self, stage1_passed: List[Dict]) -> Dict:
        """
        For each Stage-1-passing factor, try to add it to the universal pool.
        Returns action counts: {added, replaced, rejected_corr, rejected_weaker}
        """
        counts = {"added": 0, "replaced": 0, "rejected_corr": 0, "rejected_weaker": 0}
        for result in stage1_passed:
            action, _ = self.pool.add_factor(
                result,
                source_date=result.get("source_report_date"),
            )
            counts[action] = counts.get(action, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Report building and saving
    # ------------------------------------------------------------------

    def _build_report(
        self,
        all_input_factors: List[Dict],
        stage1_results: List[Dict],
        pool_actions: Dict,
        pool_summary: Dict,
    ) -> Dict:
        stage1_passed = [r for r in stage1_results if r["passed"]]
        failed_factors = []
        for r in stage1_results:
            if not r["passed"]:
                failed_factors.append({
                    "factor_id": r["factor_id"],
                    "stage_failed": 1,
                    "fail_reason": r["fail_reason"],
                    "is_sharpe": r["is_sharpe"],
                    "oos_sharpe": r["oos_sharpe"],
                })

        stage1_out = [
            {k: v for k, v in r.items() if k != "oos_equity_curve"}
            for r in stage1_results
        ]

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "train_pct": self.train_pct,
                "oos_sharpe_threshold": self.oos_sharpe_threshold,
                "sharpe_decay_threshold": self.sharpe_decay_threshold,
                "correlation_threshold": self.correlation_threshold,
                "min_oos_trades": self.min_oos_trades,
                "min_oos_days": self.min_oos_days,
            },
            "summary": {
                "total_input_factors": len(all_input_factors),
                "stage1_passed": len(stage1_passed),
                "pool_actions": pool_actions,
                "pool_state": pool_summary,
            },
            "stage1_results": stage1_out,
            "failed_factors": failed_factors,
        }

    def _save_report(self, report: Dict) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = self._output_dir / f"validation_{date_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Factor Validation Pipeline: OOS test + correlation deduplication"
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=DEFAULT_TRAIN_PCT,
        help=f"Fraction of date range used as in-sample training (default: {DEFAULT_TRAIN_PCT})",
    )
    parser.add_argument(
        "--oos-threshold",
        type=float,
        default=DEFAULT_OOS_SHARPE_THRESHOLD,
        help=f"Min OOS Sharpe ratio to pass Stage 1 (default: {DEFAULT_OOS_SHARPE_THRESHOLD})",
    )
    parser.add_argument(
        "--decay-threshold",
        type=float,
        default=DEFAULT_SHARPE_DECAY_THRESHOLD,
        help=f"Min OOS/IS Sharpe decay ratio (default: {DEFAULT_SHARPE_DECAY_THRESHOLD})",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
        help=f"Max equity curve correlation within a cluster (default: {DEFAULT_CORRELATION_THRESHOLD})",
    )
    parser.add_argument(
        "--min-oos-trades",
        type=int,
        default=DEFAULT_MIN_OOS_TRADES,
        help=f"Min number of trades in OOS period (default: {DEFAULT_MIN_OOS_TRADES})",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path to a specific daily report JSON; auto-detects latest if omitted",
    )
    parser.add_argument(
        "--last-n-days",
        type=int,
        default=None,
        metavar="N",
        help="Process the N most recent daily reports (default: latest only)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available daily reports in reports/daily_reports/",
    )
    args = parser.parse_args()

    config = {
        "train_pct": args.train_pct,
        "oos_sharpe_threshold": args.oos_threshold,
        "sharpe_decay_threshold": args.decay_threshold,
        "correlation_threshold": args.corr_threshold,
        "min_oos_trades": args.min_oos_trades,
        "report_path": args.report_path,
        "last_n_days": args.last_n_days,
        "process_all": args.all,
    }

    agent = FactorValidationAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
