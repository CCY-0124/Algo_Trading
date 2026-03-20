"""
report_generator.py

Report generator for daily backtest results.

Features:
- Generate multiple report formats (JSON, text, LLM analysis)
- Include LLM analysis summaries
- Provide recommendations and warnings
- Visualize data
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

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
    
    def generate_daily_report(self,
                             date_str: str,
                             all_results: List[Dict],
                             daily_summary: Dict,
                             performance_stats: Dict = None) -> Dict:
        """
        Generate comprehensive daily report.
        
        :param date_str: Date string (YYYY-MM-DD)
        :param all_results: All processing results
        :param daily_summary: Daily summary from context manager
        :param performance_stats: Performance statistics (optional)
        :return: Complete report dictionary
        """
        logging.info(f"Generating daily report for {date_str}")
        
        # Filter successful results
        successful_results = [r for r in all_results if r.get('status') == 'completed']
        failed_results = [r for r in all_results if r.get('status') in ['failed', 'error']]
        
        # Extract best factors
        top_factors = self._extract_top_factors(successful_results, limit=20)
        
        # Extract LLM insights
        llm_insights = self._extract_llm_insights(successful_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(successful_results, top_factors)
        
        # Build report (statistics excluded per user preference)
        report = {
            'date': date_str,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_factors_analyzed': len(all_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'promising_factors': daily_summary.get('promising_factors', 0),
                'best_sharpe': daily_summary.get('best_sharpe', 0),
                'best_factor_id': daily_summary.get('best_factor_id'),
                'total_time_seconds': performance_stats.get('session_duration_seconds', 0) if performance_stats else 0
            },
            'top_factors': top_factors,
            'llm_insights': llm_insights,
            'recommendations': recommendations,
            'performance': performance_stats,
            'all_results_count': len(all_results)
        }
        
        return report
    
    def _extract_top_factors(self, results: List[Dict], limit: int = 20) -> List[Dict]:
        """
        Extract top performing factors.

        In addition to the single best result per factor, also include all
        parameter combinations whose Sharpe ratio is greater than 1.0.
        
        :param results: Successful results (each produced by LLMOrchestrator.process_factor)
        :param limit: Maximum number of factors to return
        :return: List of top factors with per-parameter results
        """
        # Sort by Sharpe ratio of final best_result
        sorted_results = sorted(
            results,
            key=lambda x: x.get('best_result', {}).get('sharpe_ratio', 0),
            reverse=True
        )

        top_factors: List[Dict] = []

        for result in sorted_results[:limit]:
            best_result = result.get('best_result', {})
            if not best_result:
                continue

            # Collect per-parameter results from Stage 1, Stage 2, and optional optimization
            steps = result.get('steps', {})
            exploration = steps.get('exploration', {})

            param_results: List[Dict] = []

            # Stage 1 grid search results
            stage1 = exploration.get('stage1', {}) if isinstance(exploration, dict) else {}
            stage1_results = stage1.get('results', []) or []
            for r in stage1_results:
                sharpe = r.get('sharpe_ratio', 0)
                if sharpe <= 1.0:
                    continue
                param_results.append({
                    'source': 'stage1_grid_search',
                    'sharpe_ratio': sharpe,
                    'calmar_ratio': r.get('calmar_ratio', 0),
                    'total_return': r.get('total_return', 0),
                    'annual_return': r.get('annual_return', 0),
                    'max_drawdown': r.get('max_drawdown', 0),
                    'num_trades': r.get('num_trades', 0),
                    'parameters': r.get('params', {}),
                })

            # Stage 2 LLM iterative optimization results (if present)
            stage2 = exploration.get('stage2') or {}
            if isinstance(stage2, dict):
                iteration_results = stage2.get('iteration_results', []) or []
                for r in iteration_results:
                    sharpe = r.get('sharpe_ratio', 0)
                    if sharpe <= 1.0:
                        continue
                    param_results.append({
                        'source': 'stage2_iterative',
                        'sharpe_ratio': sharpe,
                        'calmar_ratio': r.get('calmar_ratio', 0),
                        'total_return': r.get('total_return', 0),
                        'annual_return': r.get('annual_return', 0),
                        'max_drawdown': r.get('max_drawdown', 0),
                        'num_trades': r.get('num_trades', 0),
                        'parameters': r.get('params', {}),
                    })

            # Optional optimization step driven by LLM suggestions
            optimization = steps.get('optimization') or {}
            if isinstance(optimization, dict):
                optimized_results = optimization.get('optimized_results', []) or []
                for r in optimized_results:
                    sharpe = r.get('sharpe_ratio', 0)
                    if sharpe <= 1.0:
                        continue
                    param_results.append({
                        'source': 'llm_guided_optimization',
                        'sharpe_ratio': sharpe,
                        'calmar_ratio': r.get('calmar_ratio', 0),
                        'total_return': r.get('total_return', 0),
                        'annual_return': r.get('annual_return', 0),
                        'max_drawdown': r.get('max_drawdown', 0),
                        'num_trades': r.get('num_trades', 0),
                        'parameters': r.get('params', {}),
                    })

            # Build factor-level summary
            factor_info = {
                'factor_id': result.get('factor_id'),
                'asset': result.get('asset'),
                'factor_name': result.get('factor_name'),
                # Final best result (for quick reference)
                'sharpe_ratio': best_result.get('sharpe_ratio', 0),
                'calmar_ratio': best_result.get('calmar_ratio', 0),
                'total_return': best_result.get('total_return', 0),
                'annual_return': best_result.get('annual_return', 0),
                'max_drawdown': best_result.get('max_drawdown', 0),
                'num_trades': best_result.get('num_trades', 0),
                'parameters': best_result.get('params', {}),
                # All parameter combinations with Sharpe > 1.0
                'param_results': param_results,
            }
            top_factors.append(factor_info)

        return top_factors
    
    def _extract_llm_insights(self, results: List[Dict]) -> Dict:
        """
        Extract LLM insights from results (counts only; no common_insights, risk_warnings, sample_analyses).
        
        :param results: Successful results
        :return: LLM insights dictionary
        """
        llm_analyses = []
        factors_with_potential = 0
        
        for result in results:
            llm_analysis = result.get('steps', {}).get('llm_analysis')
            if llm_analysis:
                llm_analyses.append(llm_analysis)
                if llm_analysis.get('has_potential', False):
                    factors_with_potential += 1
        
        return {
            'total_llm_analyses': len(llm_analyses),
            'factors_with_potential': factors_with_potential,
        }
    
    def _generate_recommendations(self, results: List[Dict], top_factors: List[Dict]) -> Dict:
        """
        Generate recommendations based on results.
        
        :param results: Successful results
        :param top_factors: Top performing factors
        :return: Recommendations dictionary
        """
        deployment_candidates = []
        needs_review = []
        
        # Deployment candidates (high Sharpe and Calmar)
        for factor in top_factors[:10]:
            sharpe = factor.get('sharpe_ratio', 0)
            calmar = factor.get('calmar_ratio', 0)
            if sharpe >= 2.0 and calmar >= 2.0:
                deployment_candidates.append({
                    'factor_id': factor.get('factor_id'),
                    'sharpe_ratio': sharpe,
                    'calmar_ratio': calmar,
                    'reason': 'High Sharpe and Calmar ratios'
                })
        
        # Needs review (moderate performance)
        for factor in top_factors[10:30]:
            sharpe = factor.get('sharpe_ratio', 0)
            if 1.0 <= sharpe < 2.0:
                needs_review.append({
                    'factor_id': factor.get('factor_id'),
                    'sharpe_ratio': sharpe,
                    'reason': 'Moderate performance, may benefit from further optimization'
                })
        
        return {
            'deployment_candidates': deployment_candidates,
            'needs_review': needs_review,
        }
    
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
    # Test the report generator
    print("Testing Report Generator...")
    
    # Create sample report
    sample_report = {
        'date': '2025-01-15',
        'summary': {
            'total_factors_analyzed': 100,
            'successful': 80,
            'failed': 20,
            'promising_factors': 25,
            'best_sharpe': 2.5,
            'best_factor_id': 'BTC_sopr'
        },
        'top_factors': [
            {
                'factor_id': 'BTC_sopr',
                'sharpe_ratio': 2.5,
                'calmar_ratio': 3.2,
                'total_return': 0.35
            }
        ],
        'llm_insights': {
            'total_llm_analyses': 25,
            'factors_with_potential': 15
        }
    }
    
    generator = ReportGenerator()
    generator.save_all_reports(sample_report, '2025-01-15')
    
    print("✅ Report generator test completed!")
