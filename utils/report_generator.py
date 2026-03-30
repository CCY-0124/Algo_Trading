"""
report_generator.py

Report generator for daily backtest results (compact JSON).
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

PROMISING_SHARPE_THRESHOLD = 1.0

_TIME_STAT_KEYS = frozenset({
    'total_time', 'avg_time', 'min_time', 'max_time', 'median_time', 'stdev_time',
})
_SECONDS_SUFFIX_KEYS = frozenset({
    'session_duration_seconds', 'total_time_seconds', 'average_operation_time',
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ReportGenerator:
    """
    Generator for daily backtest reports (JSON format).
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        :param output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.daily_reports_dir = self.output_dir / "daily_reports"
        
        self.daily_reports_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Report Generator initialized")
        logging.info(f"  Output directory: {self.output_dir}")

    @staticmethod
    def _secs_str(value: Union[int, float]) -> str:
        return f"{round(float(value), 2)} secs"

    @staticmethod
    def _round2(value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 2)
        return value

    def _final_performance_row(self, result: Dict) -> Optional[Dict]:
        best = result.get('best_result') or {}
        if not best:
            return None
        sharpe = best.get('sharpe_ratio', 0) or 0
        if sharpe < PROMISING_SHARPE_THRESHOLD:
            return None
        return {
            'factor_id': result.get('factor_id'),
            'asset': result.get('asset'),
            'factor_name': result.get('factor_name'),
            'sharpe_ratio': self._round2(sharpe),
            'calmar_ratio': self._round2(best.get('calmar_ratio', 0)),
            'total_return': self._round2(best.get('total_return', 0)),
            'annual_return': self._round2(best.get('annual_return', 0)),
            'max_drawdown': self._round2(best.get('max_drawdown', 0)),
            'num_trades': int(best.get('num_trades', 0) or 0),
            'parameters': best.get('params', {}),
        }

    def _build_promising_factors_ranked(self, successful_results: List[Dict]) -> List[Dict]:
        rows: List[Dict] = []
        for result in successful_results:
            row = self._final_performance_row(result)
            if row:
                rows.append(row)
        rows.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        return rows

    def _format_operation_stats_block(self, op_stats: Dict) -> Dict:
        out: Dict[str, Any] = {}
        for key, val in op_stats.items():
            if key in ('count', 'errors', 'operation'):
                out[key] = val
            elif key in _TIME_STAT_KEYS and isinstance(val, (int, float)):
                out[key] = self._secs_str(val)
            elif key == 'error_rate' and isinstance(val, (int, float)):
                out[key] = self._round2(val)
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                out[key] = self._round2(val)
            else:
                out[key] = val
        return out

    def _format_full_performance_report(self, raw: Dict) -> Dict:
        formatted: Dict[str, Any] = {}
        for key, val in raw.items():
            if key == 'operation_stats' and isinstance(val, dict):
                formatted[key] = {
                    name: self._format_operation_stats_block(stats)
                    for name, stats in val.items()
                    if isinstance(stats, dict)
                }
            elif key == 'bottleneck' and isinstance(val, dict):
                bn = dict(val)
                if isinstance(bn.get('stats'), dict):
                    bn['stats'] = self._format_operation_stats_block(bn['stats'])
                formatted[key] = bn
            elif key in _SECONDS_SUFFIX_KEYS and isinstance(val, (int, float)):
                formatted[key] = self._secs_str(val)
            elif key in ('total_operations', 'total_errors') and isinstance(val, int):
                formatted[key] = val
            elif key == 'summary' and isinstance(val, str):
                # JSON escapes newlines as \n in a single string; use lines array for readable reports.
                formatted['summary_lines'] = val.splitlines()
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                formatted[key] = self._round2(val)
            else:
                formatted[key] = val
        return formatted

    def _format_legacy_operation_only(self, raw: Dict) -> Dict:
        return {
            name: self._format_operation_stats_block(stats)
            for name, stats in raw.items()
            if isinstance(stats, dict)
        }

    def _format_performance_section(self, performance_stats: Optional[Dict]) -> Dict:
        if not performance_stats:
            return {}
        if 'session_duration_seconds' in performance_stats or 'operation_stats' in performance_stats:
            return self._format_full_performance_report(performance_stats)
        return self._format_legacy_operation_only(performance_stats)

    def generate_daily_report(self,
                             date_str: str,
                             all_results: List[Dict],
                             daily_summary: Dict,
                             performance_stats: Dict = None) -> Dict:
        """
        Generate compact daily report: summary, ranked promising factors (best combo only), performance.
        """
        logging.info(f"Generating daily report for {date_str}")

        successful_results = [r for r in all_results if r.get('status') == 'completed']
        failed_results = [r for r in all_results if r.get('status') in ['failed', 'error']]

        promising_ranked = self._build_promising_factors_ranked(successful_results)

        session_secs = 0.0
        if performance_stats:
            session_secs = float(performance_stats.get('session_duration_seconds', 0) or 0)

        report = {
            'date': date_str,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_factors_analyzed': len(all_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'promising_factors': daily_summary.get('promising_factors', 0),
                'best_sharpe': self._round2(daily_summary.get('best_sharpe', 0)),
                'best_factor_id': daily_summary.get('best_factor_id'),
                'total_time': self._secs_str(session_secs),
            },
            'promising_factors_ranked': promising_ranked,
            'performance': self._format_performance_section(performance_stats),
        }

        return report
    
    def save_json_report(self, report: Dict, date_str: str):
        """
        Save JSON report.
        
        :param report: Report dictionary
        :param date_str: Date string
        """
        json_file = self.daily_reports_dir / f"report_{date_str}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"JSON report saved to {json_file}")
    
    def save_all_reports(self, report: Dict, date_str: str):
        """
        Save all report formats.
        
        :param report: Report dictionary
        :param date_str: Date string
        """
        self.save_json_report(report, date_str)

        logging.info(f"All reports saved for {date_str}")


if __name__ == "__main__":
    print("Testing Report Generator...")
    gen = ReportGenerator()
    sample_results = [
        {
            'status': 'completed',
            'factor_id': 'BTC_sopr',
            'asset': 'BTC',
            'factor_name': 'sopr',
            'best_result': {
                'sharpe_ratio': 2.512345,
                'calmar_ratio': 3.2,
                'total_return': 0.35,
                'annual_return': 0.12,
                'max_drawdown': 0.08,
                'num_trades': 40,
                'params': {'rolling': 14, 'param_method': 'zscore'},
            },
        }
    ]
    summary = {
        'promising_factors': 1,
        'best_sharpe': 2.512345,
        'best_factor_id': 'BTC_sopr',
    }
    perf = {
        'session_duration_seconds': 1.234,
        'total_operations': 5,
        'total_time_seconds': 0.5,
        'total_errors': 0,
        'average_operation_time': 0.1,
        'error_rate': 0.0,
        'bottleneck': None,
        'operation_stats': {},
        'summary': 'test',
    }
    rep = gen.generate_daily_report('2025-01-15', sample_results, summary, perf)
    gen.save_all_reports(rep, '2025-01-15')
    print("Report generator test completed.")
