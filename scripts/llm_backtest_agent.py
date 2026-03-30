"""
llm_backtest_agent.py

Main agent script for daily automated backtest workflow.

Features:
- Daily automated execution
- Process all factors (900+)
- Generate daily reports
- Send notifications
- Error handling and recovery
"""

import sys
import os
import json
import logging
import argparse
import math
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
from config.trading_config import DEFAULT_INITIAL_CAPITAL, get_lot_size
from core.llm_client import LLMClient
from core.context_manager import ContextManager
from core.factor_screening import TwoStageFactorScreening
from core.llm_scheduler import IntelligentLLMScheduler
from core.data_cache import LightweightDataCache
from core.performance_monitor import PerformanceMonitor, get_monitor
from scripts.llm_orchestrator import LLMOrchestrator
from utils.local_data_loader import LocalDataLoader
from utils.report_generator import ReportGenerator

# Configure logging with verbosity control
from utils.log_config import configure_logging, set_verbosity, QuietLogger
import signal

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_logs_dir = os.path.join(_project_root, 'logs')
os.makedirs(_logs_dir, exist_ok=True)
log_file = os.path.join(_logs_dir, f'llm_agent_{datetime.now().strftime("%Y%m%d")}.log')

# Default verbosity: 1 (normal)
verbosity = 1


class LLMBacktestAgent:
    """
    Main agent for daily automated backtest workflow.
    
    Responsibilities:
    - Initialize all components
    - Process all factors
    - Generate reports
    - Handle errors and recovery
    """
    
    def __init__(self, config: Dict):
        """
        Initialize LLM backtest agent.
        
        :param config: Configuration dictionary
        """
        self.config = config
        self.date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Initialize components
        self._initialize_components()
        
        logging.info("LLM Backtest Agent initialized")
        logging.info(f"Date: {self.date_str}")
    
    def _initialize_components(self):
        """Initialize all system components."""
        logging.info("Initializing components...")
        
        # Strategy (defaults align with config.trading_config when keys omitted)
        _assets = self.config.get('assets') or ['BTC']
        _primary_asset = _assets[0] if _assets else 'BTC'
        self.strategy = EnhancedNonPriceStrategy(
            initial_capital=self.config.get('initial_capital', DEFAULT_INITIAL_CAPITAL),
            min_lot_size=self.config.get('min_lot_size', get_lot_size(_primary_asset)),
            use_api=self.config.get('use_api', True),
            param_method=self.config.get('param_method', 'pct_change')
        )
        
        # LLM Client (Jetson primary; Mac mini fallback when available)
        llm_config = self.config.get('llm', {})
        self.llm_client = LLMClient(
            api_url=llm_config.get('api_url', 'http://192.168.1.212:11434/api/chat'),
            model=llm_config.get('model', 'qwen3.5:2b'),
            timeout=int(llm_config.get('timeout', 120)),
            fallback_urls=llm_config.get('fallback_urls', []),
            wall_clock_limit=int(llm_config.get('wall_clock_limit', 300)),
            stream_stuck_seconds=int(llm_config.get('stream_stuck_seconds', 60)),
        )
        
        # Context Manager
        self.context_manager = ContextManager()
        
        # Performance Monitor
        self.performance_monitor = get_monitor()
        
        # Data Cache
        self.data_cache = LightweightDataCache(
            max_size=self.config.get('cache_size', 50)
        )
        
        # Two-Stage Screening
        self.screening = TwoStageFactorScreening(
            strategy=self.strategy,
            llm_client=self.llm_client,
            min_sharpe_threshold=self.config.get('min_sharpe_threshold', 0.5),
            min_calmar_threshold=self.config.get('min_calmar_threshold', 0.3),
            performance_monitor=self.performance_monitor
        )
        
        # LLM Scheduler
        self.scheduler = IntelligentLLMScheduler(
            llm_client=self.llm_client,
            min_sharpe_threshold=self.config.get('llm_min_sharpe', 1.0),
            min_calmar_threshold=self.config.get('llm_min_calmar', 0.5),
            max_optimization_rounds=self.config.get('max_optimization_rounds', 3),
            batch_size=self.config.get('batch_size', 10),
            performance_monitor=self.performance_monitor
        )
        
        # Intelligent Parameter Generator
        from core.intelligent_param_generator import IntelligentParamGenerator
        self.param_generator = IntelligentParamGenerator(
            llm_client=self.llm_client,
            target_combinations=self.config.get('target_param_combinations', 500)
        )
        
        # LLM Orchestrator
        self.orchestrator = LLMOrchestrator(
            strategy=self.strategy,
            llm_client=self.llm_client,
            context_manager=self.context_manager,
            screening=self.screening,
            scheduler=self.scheduler,
            data_cache=self.data_cache,
            performance_monitor=self.performance_monitor,
            param_generator=self.param_generator
        )
        
        # Data Loader
        data_path = self.config.get('data_path', r"D:\Trading_Data\glassnode_data2")
        self.data_loader = LocalDataLoader(base_path=data_path)
        
        # Report Generator
        self.report_generator = ReportGenerator()
        
        logging.info("All components initialized")
    
    def test_llm_connection(self) -> bool:
        """
        Test connection to LLM service.
        
        :return: True if connection successful
        """
        logging.info("Testing LLM connection...")
        success = self.llm_client.test_connection()
        
        if success:
            logging.info("LLM connection successful")
        else:
            logging.error("LLM connection failed")
            logging.error("Please ensure Ollama is running on Jetson")
        
        return success
    
    def get_factor_list(self) -> List[Dict]:
        """
        Get list of factors to analyze.

        Priority order:
        1. config 'factors' list (explicit names) - verified against disk
        2. metrics_info_{asset}.csv in data_path - only includes factors with CSV on disk
        3. Fallback: disk scan via get_available_factors()

        :return: List of factor dictionaries with 'asset' and 'factor_name' keys
        """
        assets = self.config.get('assets', ['BTC'])
        factors = self.config.get('factors', [])
        max_factors = self.config.get('max_factors', None)
        data_path = self.config.get('data_path', r'D:\Trading_Data\glassnode_data2')

        factor_list = []

        for asset in assets:
            if factors:
                # Explicit list from config - verify each CSV exists on disk
                asset_dir = os.path.join(data_path, asset)
                for factor in factors:
                    csv_path = os.path.join(asset_dir, f"{factor}.csv")
                    if not os.path.exists(csv_path):
                        logging.warning(f"Factor {asset}/{factor} in config has no CSV on disk, skipping.")
                        continue
                    factor_list.append({'asset': asset, 'factor_name': factor})
                    if max_factors and len(factor_list) >= max_factors:
                        break
            else:
                # Try metrics_info as primary source
                metrics_file = os.path.join(data_path, f"metrics_info_{asset.lower()}.csv")
                if os.path.exists(metrics_file):
                    available_factors = self._get_factors_from_metrics_info(asset, metrics_file, data_path)
                    logging.info(f"Using metrics_info as factor source for {asset}: {len(available_factors)} factors available")
                else:
                    # Fallback: disk scan
                    logging.warning(f"No metrics_info_{asset.lower()}.csv found in {data_path}, falling back to disk scan.")
                    try:
                        available_factors = self.data_loader.get_available_factors(asset)
                    except Exception as e:
                        logging.error(f"Error getting factors for {asset}: {e}")
                        available_factors = []

                for factor in available_factors:
                    factor_list.append({'asset': asset, 'factor_name': factor})
                    if max_factors and len(factor_list) >= max_factors:
                        break

            if max_factors and len(factor_list) >= max_factors:
                break

        logging.info(f"Found {len(factor_list)} factors to analyze")
        return factor_list

    def _get_factors_from_metrics_info(self, asset: str, metrics_file: str, data_path: str) -> List[str]:
        """
        Extract factor names from metrics_info CSV, using the same filename convention as daily_download.py.
        Only returns factors whose CSV file exists on disk right now.

        Filename convention: join path parts with '_' + '_tier{tier}.csv'
        Example: /addresses/accumulation_balance, tier=3 -> addresses_accumulation_balance_tier3.csv

        :param asset: Asset symbol (e.g., 'BTC')
        :param metrics_file: Full path to metrics_info_{asset}.csv
        :param data_path: Base data directory
        :return: Sorted list of factor names (without .csv extension)
        """
        import pandas as pd

        try:
            df = pd.read_csv(metrics_file)
        except Exception as e:
            logging.error(f"Failed to read {metrics_file}: {e}")
            return []

        required_cols = ['path', 'tier']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logging.error(f"metrics_info missing required columns {missing}: {metrics_file}")
            return []

        asset_dir = os.path.join(data_path, asset)
        factors = []
        skipped = 0

        for _, row in df.iterrows():
            metric_path = str(row['path'])
            tier = int(row.get('tier', 1))

            # Check supported_assets when the column exists
            if 'supported_assets' in df.columns:
                supported_str = str(row.get('supported_assets', asset))
                supported = [s.strip().upper() for s in supported_str.split(',') if s.strip()]
                if asset.upper() not in supported:
                    continue

            # Build factor name using same convention as daily_download.py
            parts = [p for p in metric_path.strip('/').split('/') if p]
            factor_name = f"{'_'.join(parts)}_tier{tier}"

            # Only include if CSV is present on disk right now
            csv_path = os.path.join(asset_dir, f"{factor_name}.csv")
            if os.path.exists(csv_path):
                factors.append(factor_name)
            else:
                skipped += 1

        logging.info(f"metrics_info -> {len(factors)} factors with CSV on disk, {skipped} skipped (no CSV)")
        return sorted(factors)
    
    def process_all_factors(
        self,
        factor_list: List[Dict],
        start_index: int = 0,
        initial_results: Optional[List[Dict]] = None,
        factor_list_full: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Process all factors through complete workflow.

        :param factor_list: List of factors to process (may be full list or remaining when resuming)
        :param start_index: 0-based index to start from (used when not resuming by key)
        :param initial_results: Results from previous run when resuming (each has asset, factor_name)
        :param factor_list_full: Full factor list for progress file and counts; default factor_list
        :return: List of processing results
        """
        total_count = len(factor_list)
        full_list = factor_list_full if factor_list_full is not None else factor_list
        total_full = len(full_list)
        self._stopped_early = False
        results = list(initial_results) if initial_results else []
        successful = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') in ('failed', 'error'))
        skipped = len(results) - successful - failed

        if initial_results:
            logging.info(f"Resuming: {len(results)} already done, {total_count} remaining (total run {total_full})")
        elif start_index > 0:
            logging.info(f"Resuming from factor {start_index + 1}/{total_full} ({len(results)} already done)")
        else:
            logging.info(f"Starting to process {total_full} factors")
        
        # Get parameter grid from config (if use_intelligent_params is False)
        use_intelligent_params = self.config.get('use_intelligent_params', True)
        
        if use_intelligent_params:
            # Will be generated intelligently by param_generator
            param_grid = None
            logging.info("Using intelligent parameter generation (AI will decide parameters)")
        else:
            # Use config-based parameter grid
            param_space = self.config.get('parameter_space', {})
            param_grid = {
                'rolling': param_space.get('rolling_windows', [1, 5, 10, 20]),
                'long_param': param_space.get('long_params', None) or 
                             list(range(-20, 1, 2)) if param_space.get('long_params') is None else param_space.get('long_params'),
                'short_param': param_space.get('short_params', None) or 
                              list(range(0, 21, 2)) if param_space.get('short_params') is None else param_space.get('short_params')
            }
            
            # Convert to proper format if needed
            # Handle long_param (from config: long_params)
            if 'long_params' in param_grid and param_grid['long_params'] is not None:
                if isinstance(param_grid['long_params'][0], int):
                    param_grid['long_param'] = [x / 100.0 for x in param_grid['long_params']]
                else:
                    param_grid['long_param'] = param_grid['long_params']
            else:
                param_grid['long_param'] = param_grid.get('long_param', [-0.04])
            
            # Handle short_param (from config: short_params)
            if 'short_params' in param_grid and param_grid['short_params'] is not None:
                if isinstance(param_grid['short_params'][0], int):
                    param_grid['short_param'] = [x / 100.0 for x in param_grid['short_params']]
                else:
                    param_grid['short_param'] = param_grid['short_params']
            else:
                param_grid['short_param'] = param_grid.get('short_param', [0.14])
        
        # Calculate date range: last 3 years if not specified
        config_date_range = self.config.get('date_range')
        if config_date_range and len(config_date_range) == 2:
            date_range = tuple(config_date_range)
        else:
            # Auto-calculate: last 3 years from today
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            logging.info(f"Using auto-calculated date range (last 3 years): {date_range[0]} to {date_range[1]}")
        
        offset = len(results)
        for idx in range(start_index, total_count):
            factor_info = factor_list[idx]
            asset = factor_info['asset']
            factor_name = factor_info['factor_name']
            i = offset + (idx - start_index) + 1

            logging.info(f"\n{'='*80}")
            logging.info(f"Processing factor {i}/{total_full}: {asset}/{factor_name}")
            logging.info(f"{'='*80}")

            try:
                result = self.orchestrator.process_factor(
                    asset=asset,
                    factor_name=factor_name,
                    param_grid=param_grid,
                    date_range=date_range,
                    use_llm_optimization=self.config.get('use_llm_optimization', True)
                )

                results.append(result)

                if result.get('status') == 'completed':
                    successful += 1
                    best_sharpe = result.get('best_result', {}).get('sharpe_ratio', 0)
                    logging.info(f"[OK] Completed: Sharpe {best_sharpe:.4f}")
                elif result.get('status') == 'failed':
                    failed += 1
                    logging.warning(f"[FAILED] Failed: {result.get('reason', 'Unknown')}")
                else:
                    skipped += 1
                    logging.info(f"[SKIPPED] Skipped: {result.get('reason', 'Unknown')}")

                self._log_backtest_record(result)
                self._save_progress(results, i, total_full, full_list)

            except Exception as e:
                logging.error(f"Error processing {asset}/{factor_name}: {e}")
                import traceback
                traceback.print_exc()

                results.append({
                    'asset': asset,
                    'factor_name': factor_name,
                    'status': 'error',
                    'error': str(e)
                })
                failed += 1
                self._log_backtest_record(results[-1])
                self._save_progress(results, i, total_full, full_list)
                self._stopped_early = True
                logging.warning(
                    "Stopping run due to error (e.g. LLM timeout, file, or connection). "
                    "Report will be generated from completed results so far."
                )
                break
        
        # Final summary
        from utils.log_config import should_log
        
        if should_log('info'):
            logging.info(f"\n{'='*80}")
            logging.info("PROCESSING SUMMARY")
            logging.info(f"{'='*80}")
            logging.info(f"Total factors: {total_full}")
            logging.info(f"Successful: {successful}")
            logging.info(f"Failed: {failed}")
            logging.info(f"Skipped: {skipped}")
            logging.info(f"{'='*80}")

        if initial_results and factor_list_full is not None:
            completed_map = {(r.get('asset'), r.get('factor_name')): r for r in initial_results}
            new_results = results[len(initial_results):]
            new_map = {(r.get('asset'), r.get('factor_name')): r for r in new_results}
            return [
                completed_map.get((f['asset'], f['factor_name'])) or new_map.get((f['asset'], f['factor_name']))
                for f in factor_list_full
            ]
        return results
    
    def _get_progress_path(self) -> str:
        """Return path to progress file. Single file so resume works across days."""
        return "reports/progress_latest.json"

    def _load_progress(self) -> Optional[tuple]:
        """
        Load progress from last run if available (any day).
        Each result in progress includes asset and factor_name (coin and factor).

        :return: (results, completed_set, factor_list_snapshot) or None.
                 completed_set: set of (asset, factor_name) already done.
                 factor_list_snapshot: list used in that run, or None.
        """
        progress_file = self._get_progress_path()
        if not os.path.exists(progress_file):
            return None
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results = data.get('results', [])
            if not results:
                return None
            completed_set = {(r.get('asset'), r.get('factor_name')) for r in results if r.get('asset') and r.get('factor_name')}
            factor_list_snapshot = data.get('factor_list')
            return (results, completed_set, factor_list_snapshot)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load progress file: {e}")
            return None

    def _save_progress(
        self,
        results: List[Dict],
        current: int,
        total: int,
        factor_list: Optional[List[Dict]] = None
    ):
        """
        Save progress periodically. Written to a single file so resume works any day.
        Each result in results includes asset and factor_name (coin and factor).

        :param results: Current results (each has asset, factor_name, status, etc.)
        :param current: Current progress count
        :param total: Total factors
        :param factor_list: Full factor list for this run (optional, for reference)
        """
        progress_file = self._get_progress_path()
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)

        def _strip_equity_curves(obj):
            if isinstance(obj, dict):
                return {k: (_strip_equity_curves(v) if k != 'equity_curve' else []) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_strip_equity_curves(i) for i in obj]
            return obj

        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'current': current,
            'total': total,
            'progress_pct': (current / total) * 100 if total else 0,
            'results': _strip_equity_curves(results),
            'factor_list': factor_list
        }

        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, default=str)

        logging.info(f"Progress saved: {current}/{total} ({progress_data['progress_pct']:.1f}%)")

    def _log_backtest_record(self, result: Dict):
        """
        Log one structured run record per factor with prefix [BACKTEST_RECORD] so it can
        be filtered (e.g. grep) to get a third file: backtest date/time, asset, factor,
        method, reason, grid, stage2 or not, stage2 summary, final result.
        Uses INFO level so it appears in the main log; filter by [BACKTEST_RECORD] to extract.
        """
        try:
            steps = result.get('steps', {})
            exploration = steps.get('exploration', {})
            stage1 = exploration.get('stage1') or {}
            stage2 = exploration.get('stage2')
            param_gen = steps.get('param_generation', {})
            best = result.get('best_result') or {}
            params = best.get('params', {})

            grid_combos = param_gen.get('combinations')
            grid_summary = str(grid_combos) + " combos" if grid_combos is not None else "N/A"

            stage2_used = stage2 is not None
            stage2_summary = ""
            stage2_reasons = []
            if stage2:
                iters = stage2.get('iterations', 0)
                imp = stage2.get('improvement')
                stage2_summary = f"iterations={iters}, improvement={imp:.4f}" if imp is not None else f"iterations={iters}"
                for ia in stage2.get('llm_analyses', [])[:2]:
                    r = ia.get('reason') or ia.get('rolling_window_suggestion', {}).get('reason')
                    if r:
                        stage2_reasons.append(str(r)[:80])

            record = {
                "timestamp": result.get('timestamp') or datetime.now().isoformat(),
                "asset": result.get('asset', ""),
                "factor": result.get('factor_name', ""),
                "param_method": params.get('param_method', ""),
                "method_reason": param_gen.get('method_reason', 'N/A'),
                "grid_summary": grid_summary,
                "stage2_used": stage2_used,
                "stage2_summary": stage2_summary,
                "stage2_reasons": stage2_reasons[:2] if stage2_reasons else [],
                "final_status": result.get('status', ""),
                "final_sharpe": best.get('sharpe_ratio'),
                "final_calmar": best.get('calmar_ratio'),
                "final_reason": result.get('reason') or result.get('error', ''),
            }
            logging.info("[BACKTEST_RECORD] %s", json.dumps(record, ensure_ascii=False, default=str))
        except Exception:
            logging.info("[BACKTEST_RECORD] %s", json.dumps({"error": "record build failed", "factor": result.get("factor_name", "")}, ensure_ascii=False, default=str))

    def generate_daily_summary(self, all_results: List[Dict]) -> Dict:
        """
        Generate daily summary for context manager.
        
        :param all_results: All processing results
        :return: Summary dictionary
        """
        # Filter successful results
        successful_results = [r for r in all_results if r.get('status') == 'completed']
        
        if not successful_results:
            return {
                'total_factors': len(all_results),
                'promising_factors': 0,
                'best_factor_id': None,
                'best_sharpe': 0.0,
                'summary_data': {}
            }
        
        # Find best factor
        best_result = max(
            successful_results,
            key=lambda x: x.get('best_result', {}).get('sharpe_ratio', 0)
        )
        
        best_sharpe = best_result.get('best_result', {}).get('sharpe_ratio', 0)
        promising_count = len([r for r in successful_results 
                              if r.get('best_result', {}).get('sharpe_ratio', 0) >= 1.0])
        
        summary = {
            'total_factors': len(all_results),
            'promising_factors': promising_count,
            'best_factor_id': best_result.get('factor_id'),
            'best_sharpe': best_sharpe,
            'summary_data': {
                'successful': len(successful_results),
                'failed': len(all_results) - len(successful_results),
                'top_10_sharpe': sorted(
                    [r.get('best_result', {}).get('sharpe_ratio', 0) 
                     for r in successful_results],
                    reverse=True
                )[:10]
            }
        }
        
        return summary

    def _finish_workflow(self, all_results: List[Dict], partial: bool = False) -> Dict:
        """
        Generate summary, save context, generate and save reports. Used on normal
        completion or after stopping due to error (with partial results).
        """
        if not all_results:
            logging.warning("No results to report.")
            return {
                'status': 'error' if partial else 'success',
                'date': self.date_str,
                'error': 'No results' if partial else None,
                'summary': None,
                'report': None,
                'results': []
            }
        summary = self.generate_daily_summary(all_results)
        self.context_manager.save_daily_summary(
            date=self.date_str,
            total_factors=summary['total_factors'],
            promising_factors=summary['promising_factors'],
            best_factor_id=summary['best_factor_id'],
            best_sharpe=summary['best_sharpe'],
            summary_data=summary.get('summary_data', {})
        )
        self.performance_monitor.print_report()
        perf_report = self.performance_monitor.generate_performance_report()
        daily_report = self.report_generator.generate_daily_report(
            date_str=self.date_str,
            all_results=all_results,
            daily_summary=summary,
            performance_stats=perf_report
        )
        self.report_generator.save_all_reports(daily_report, self.date_str)
        if self.data_cache:
            self.data_cache.print_stats()
        if partial:
            logging.info(
                f"\n{'='*80}\nWORKFLOW STOPPED (error/partial). Report generated from {len(all_results)} results.\n"
                f"Reports saved to: reports/daily_reports/\n{'='*80}"
            )
        else:
            logging.info(f"\n{'='*80}")
            logging.info(f"DAILY WORKFLOW COMPLETED - {self.date_str}")
            logging.info(f"{'='*80}")
            logging.info(f"Best Sharpe: {summary['best_sharpe']:.4f}")
            logging.info(f"Promising Factors: {summary['promising_factors']}/{summary['total_factors']}")
            logging.info(f"Reports saved to: reports/daily_reports/")
        return {
            'status': 'partial' if partial else 'success',
            'date': self.date_str,
            'summary': summary,
            'report': daily_report,
            'results': all_results
        }

    def _finish_workflow_on_error(self, error: Exception) -> Dict:
        """
        On workflow exception, try to load progress and generate report from
        partial results; then exit cleanly.
        """
        logging.warning("Attempting to generate report from saved progress (if any).")
        checkpoint = self._load_progress()
        if checkpoint:
            partial_results, _, _ = checkpoint
            if partial_results:
                logging.info(f"Loaded {len(partial_results)} results from progress file.")
                return self._finish_workflow(partial_results, partial=True)
        logging.error("No progress file or results available. Exiting without report.")
        return {
            'status': 'error',
            'date': self.date_str,
            'error': str(error)
        }

    def _run_validation(self, report_path: str = None):
        """
        Call FactorValidationAgent silently; print a one-line summary when done.
        Returns the report dict (or None on error).
        """
        import importlib
        fva_mod = importlib.import_module("scripts.factor_validation_agent")
        FactorValidationAgent = fva_mod.FactorValidationAgent

        val_config = {
            "train_pct": self.config.get("validation", {}).get("train_pct", 0.65),
            "oos_sharpe_threshold": self.config.get("validation", {}).get("oos_sharpe_threshold", 0.8),
            "sharpe_decay_threshold": self.config.get("validation", {}).get("sharpe_decay_threshold", 0.4),
            "correlation_threshold": self.config.get("validation", {}).get("correlation_threshold", 0.7),
            "min_oos_trades": self.config.get("validation", {}).get("min_oos_trades", 5),
            "min_oos_days": self.config.get("validation", {}).get("min_oos_days", 90),
            "report_path": report_path,
        }
        val_agent = FactorValidationAgent(val_config)

        # Suppress all output during validation; print summary after
        logging.disable(logging.CRITICAL)
        _fd1 = os.dup(1)
        _fd2 = os.dup(2)
        _dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_dn, 1)
        os.dup2(_dn, 2)
        os.close(_dn)
        _orig_out = sys.stdout
        _orig_err = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        report = None
        try:
            report = val_agent.run()
        finally:
            logging.disable(logging.NOTSET)
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = _orig_out
            sys.stderr = _orig_err
            os.dup2(_fd1, 1)
            os.dup2(_fd2, 2)
            os.close(_fd1)
            os.close(_fd2)

        if report:
            s = report.get("summary", {})
            pa = s.get("pool_actions", {})
            ps = s.get("pool_state", {})
            passed = s.get("stage1_passed", 0)
            total = s.get("total_input_factors", 0)
            print(
                f"  Validation: {passed}/{total} passed | "
                f"added={pa.get('added',0)} replaced={pa.get('replaced',0)} "
                f"rejected={pa.get('rejected_corr',0)+pa.get('rejected_weaker',0)} | "
                f"pool={ps.get('total',0)}"
            )
        return report

    def run_daily_workflow(self, resume: bool = False, validate: bool = False):
        """
        Run complete daily workflow.

        :param resume: If True, load progress from last run and continue from there.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"STARTING DAILY WORKFLOW - {self.date_str}")
        logging.info(f"{'='*80}\n")

        try:
            if not self.test_llm_connection():
                logging.warning("LLM connection failed, but continuing...")

            factor_list = self.get_factor_list()

            if not factor_list:
                logging.error("No factors to process!")
                return {'status': 'error', 'date': self.date_str, 'error': 'No factors to process'}

            start_index = 0
            initial_results = None

            if resume:
                checkpoint = self._load_progress()
                if checkpoint:
                    initial_results, completed_set, _ = checkpoint
                    remaining = [f for f in factor_list if (f.get('asset'), f.get('factor_name')) not in completed_set]
                    if not remaining:
                        logging.info(
                            f"Resume: all {len(factor_list)} factors already done in checkpoint. Nothing left to process."
                        )
                        all_results = initial_results
                    else:
                        logging.info(
                            f"Resume: loaded {len(initial_results)} results (by asset/factor), "
                            f"{len(remaining)} factors remaining."
                        )
                        all_results = self.process_all_factors(
                            remaining,
                            initial_results=initial_results,
                            factor_list_full=factor_list
                        )
                else:
                    logging.info("Resume requested but no progress file found. Starting from beginning.")
                    all_results = self.process_all_factors(factor_list)
            else:
                all_results = self.process_all_factors(factor_list)

            result = self._finish_workflow(all_results, partial=getattr(self, '_stopped_early', False))

            if validate and result.get("status") in ("success", "partial"):
                self._run_validation()

            return result

        except Exception as e:
            logging.error(f"Error in daily workflow: {e}")
            import traceback
            traceback.print_exc()
            return self._finish_workflow_on_error(e)

    def run_batch_validate_loop(self, batch_size: int, validate: bool = True):
        """
        Process all factors in batches until all are done.
        - Shows per-factor progress on console; suppresses backtest noise.
        - Saves a batch snapshot report after each batch.
        - Runs validation on each batch's results and updates the pool.
        - Press Ctrl+C to stop gracefully after the current factor and write daily report.
        - Resumes from progress_latest.json if a previous run was interrupted.
        """
        self._stop_requested = False

        def _handle_stop(signum, frame):
            # Write directly to stderr so it shows even when stdout is suppressed
            import sys as _sys
            _sys.stderr.write("\n[Stopping after current factor... daily report will be written]\n")
            _sys.stderr.flush()
            self._stop_requested = True

        original_handler = signal.signal(signal.SIGINT, _handle_stop)

        batch_reports_dir = Path(_project_root) / "reports" / "batch_reports"
        batch_reports_dir.mkdir(parents=True, exist_ok=True)

        try:
            factor_list = self.get_factor_list()
            if not factor_list:
                print("No factors to process!")
                return

            checkpoint = self._load_progress()
            completed_set = set()
            accumulated_results = []
            if checkpoint:
                accumulated_results, completed_set, _ = checkpoint
                print(f"Resuming: {len(completed_set)}/{len(factor_list)} factors already done")

            remaining = [
                f for f in factor_list
                if (f.get("asset"), f.get("factor_name")) not in completed_set
            ]
            total_batches = math.ceil(len(remaining) / batch_size) if remaining else 0
            print(f"Starting: {len(remaining)} factors | {total_batches} batches of {batch_size}")

            batch_num = 0
            while remaining and not self._stop_requested:
                batch_num += 1
                batch = remaining[:batch_size]
                remaining = remaining[batch_size:]

                print(f"\n=== Batch {batch_num}/{total_batches} ({len(batch)} factors) ===")

                # Process one factor at a time: show progress, check stop flag
                for i, factor_info in enumerate(batch):
                    if self._stop_requested:
                        break
                    left = len(batch) - i - 1
                    print(
                        f"  [{i+1}/{len(batch)}] {factor_info['asset']}/{factor_info['factor_name']}"
                        f"  |  {left} left in batch",
                        flush=True,
                    )
                    # Suppress all output: redirect OS-level fds so even cached
                    # StreamHandler references (which ignore sys.stdout reassignment)
                    # get silenced. Also disable logging globally as extra measure.
                    import sys as _sys
                    logging.disable(logging.CRITICAL)
                    _orig_stdout_obj = _sys.stdout
                    _orig_stderr_obj = _sys.stderr
                    _fd1 = os.dup(1)   # save original stdout fd
                    _fd2 = os.dup(2)   # save original stderr fd
                    _dn = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(_dn, 1)
                    os.dup2(_dn, 2)
                    os.close(_dn)
                    _sys.stdout = open(os.devnull, 'w')
                    _sys.stderr = open(os.devnull, 'w')
                    try:
                        result = self.process_all_factors(
                            [factor_info],
                            initial_results=accumulated_results,
                            factor_list_full=factor_list,
                        )
                    finally:
                        logging.disable(logging.NOTSET)
                        _sys.stdout.close()
                        _sys.stderr.close()
                        _sys.stdout = _orig_stdout_obj
                        _sys.stderr = _orig_stderr_obj
                        os.dup2(_fd1, 1)
                        os.dup2(_fd2, 2)
                        os.close(_fd1)
                        os.close(_fd2)
                    accumulated_results = [r for r in result if r is not None]
                    self.strategy.data_loader.clear_factor_cache()

                # Save batch snapshot (only this batch's factors)
                batch_keys = {(f["asset"], f["factor_name"]) for f in batch}
                batch_only = [
                    r for r in accumulated_results
                    if r and (r.get("asset"), r.get("factor_name")) in batch_keys
                ]
                batch_summary = self.generate_daily_summary(batch_only)
                batch_report = self.report_generator.generate_daily_report(
                    date_str=self.date_str,
                    all_results=batch_only,
                    daily_summary=batch_summary,
                    performance_stats={},
                )
                batch_file = batch_reports_dir / f"report_{self.date_str}_batch{batch_num:03d}.json"
                with open(batch_file, "w", encoding="utf-8") as f:
                    json.dump(batch_report, f, indent=2, ensure_ascii=False, default=str)
                print(f"  Batch report saved -> {batch_file.name}")

                if validate:
                    try:
                        self._run_validation(report_path=str(batch_file))
                    except Exception as e:
                        print(f"  Validation failed: {e}")

            # Write final daily report
            valid = [r for r in accumulated_results if r is not None]
            if valid:
                stopped_early = self._stop_requested or bool(remaining)
                print(f"\nWriting daily report ({len(valid)} factors processed)...")
                self._finish_workflow(valid, partial=stopped_early)
                print(f"Daily report saved -> reports/daily_reports/report_{self.date_str}.json")

            if self._stop_requested:
                print("Stopped by user.")
            elif not remaining:
                print("All batches complete!")

        finally:
            signal.signal(signal.SIGINT, original_handler)


def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from file or use defaults.
    
    :param config_path: Path to config file
    :return: Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Default configuration
    return {
        'assets': ['BTC'],
        'factors': [],  # Empty = all available factors
        'initial_capital': DEFAULT_INITIAL_CAPITAL,
        'min_lot_size': get_lot_size('BTC'),
        'use_api': True,
        'param_method': 'pct_change',
        'date_range': ['2023-01-01', '2025-08-10'],
        'parameter_space': {
            'rolling_windows': [1, 5, 10, 20],
            'long_params': None,  # Will use default ranges
            'short_params': None
        },
        'min_sharpe_threshold': 0.5,
        'min_calmar_threshold': 0.3,
        'llm_min_sharpe': 1.0,
        'llm_min_calmar': 0.5,
        'max_optimization_rounds': 3,
        'batch_size': 10,
        'cache_size': 50,
        'use_llm_optimization': True,
        'llm': {
            'api_url': 'http://192.168.1.212:11434/api/chat',
            'model': 'qwen3.5:2b',
            'fallback_urls': []
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='LLM Backtest Agent')
    parser.add_argument(
        '--config',
        type=str,
        default='config/llm_agent_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test-llm',
        action='store_true',
        help='Test LLM connection and exit'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (use -v, -vv, -vvv)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (only errors)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last saved progress (any day; uses reports/progress_latest.json)'
    )
    parser.add_argument(
        '--max-factors',
        type=int,
        default=None,
        metavar='N',
        help='Limit run to first N factors (overrides config max_factors)'
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run FactorValidationAgent after daily workflow completes",
    )
    parser.add_argument(
        "--batch-validate",
        type=int,
        default=None,
        metavar="N",
        help="Process factors in batches of N, validating after each batch",
    )
    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=None,
        metavar="ASSET",
        help="Assets to backtest, e.g. --assets BTC ETH (overrides config)",
    )

    args = parser.parse_args()
    
    # Set verbosity level
    if args.quiet:
        verbosity = 0
    elif args.verbose == 0:
        verbosity = 1  # Normal
    elif args.verbose == 1:
        verbosity = 2  # Verbose
    else:
        verbosity = 3  # Debug
    
    # Configure logging
    configure_logging(log_file=log_file, verbosity=verbosity, use_progress=True)
    logging.info(f"Log file: {os.path.abspath(log_file)}")

    # Load configuration
    config = load_config(args.config)
    if args.max_factors is not None:
        config['max_factors'] = args.max_factors
        logging.info(f"Limiting to {args.max_factors} factors (from --max-factors)")
    if args.assets is not None:
        config['assets'] = args.assets
        logging.info(f"Assets overridden to: {args.assets}")

    # Create agent
    agent = LLMBacktestAgent(config)
    
    # Test LLM connection if requested
    if args.test_llm:
        success = agent.test_llm_connection()
        sys.exit(0 if success else 1)
    
    if args.batch_validate:
        agent.run_batch_validate_loop(batch_size=args.batch_validate, validate=True)
        sys.exit(0)

    result = agent.run_daily_workflow(resume=args.resume, validate=args.validate)

    status = result.get('status', 'error')
    if status == 'success':
        logging.info("[OK] Daily workflow completed successfully!")
        sys.exit(0)
    if status == 'partial':
        logging.warning("[PARTIAL] Workflow stopped due to error. Report generated from completed results.")
        sys.exit(0)
    logging.error("[ERROR] Daily workflow failed with no report.")
    sys.exit(1)


if __name__ == "__main__":
    main()
