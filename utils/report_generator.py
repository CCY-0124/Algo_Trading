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
    Generator for daily backtest reports.
    
    Generates:
    - JSON reports (machine-readable)
    - Text reports (human-readable)
    - LLM analysis reports (separate file)
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        :param output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.daily_reports_dir = self.output_dir / "daily_reports"
        self.llm_analysis_dir = self.output_dir / "llm_analysis"
        
        # Create directories
        self.daily_reports_dir.mkdir(parents=True, exist_ok=True)
        self.llm_analysis_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Calculate statistics
        statistics = self._calculate_statistics(successful_results)
        
        # Build report
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
            'statistics': statistics,
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
        
        :param results: Successful results
        :param limit: Maximum number of factors to return
        :return: List of top factors
        """
        # Sort by Sharpe ratio
        sorted_results = sorted(
            results,
            key=lambda x: x.get('best_result', {}).get('sharpe_ratio', 0),
            reverse=True
        )
        
        top_factors = []
        for result in sorted_results[:limit]:
            best_result = result.get('best_result', {})
            if best_result:
                factor_info = {
                    'factor_id': result.get('factor_id'),
                    'asset': result.get('asset'),
                    'factor_name': result.get('factor_name'),
                    'sharpe_ratio': best_result.get('sharpe_ratio', 0),
                    'calmar_ratio': best_result.get('calmar_ratio', 0),
                    'total_return': best_result.get('total_return', 0),
                    'annual_return': best_result.get('annual_return', 0),
                    'max_drawdown': best_result.get('max_drawdown', 0),
                    'win_rate': best_result.get('win_rate', 0),
                    'num_trades': best_result.get('num_trades', 0),
                    'parameters': best_result.get('params', {}),
                    'source': result.get('best_source', 'unknown')
                }
                top_factors.append(factor_info)
        
        return top_factors
    
    def _extract_llm_insights(self, results: List[Dict]) -> Dict:
        """
        Extract LLM insights from results.
        
        :param results: Successful results
        :return: LLM insights dictionary
        """
        llm_analyses = []
        factors_with_potential = 0
        common_insights = []
        risk_warnings = []
        
        for result in results:
            llm_analysis = result.get('steps', {}).get('llm_analysis')
            if llm_analysis:
                llm_analyses.append(llm_analysis)
                
                if llm_analysis.get('has_potential', False):
                    factors_with_potential += 1
                
                # Extract risk warnings
                risk_warning = llm_analysis.get('risk_warning')
                if risk_warning:
                    risk_warnings.append(risk_warning)
        
        # Extract common patterns (simplified)
        if llm_analyses:
            # Count parameter suggestions
            rolling_suggestions = {}
            for analysis in llm_analyses:
                rolling = analysis.get('rolling_window_suggestion', {}).get('recommended')
                if rolling:
                    rolling_suggestions[rolling] = rolling_suggestions.get(rolling, 0) + 1
            
            if rolling_suggestions:
                most_common_rolling = max(rolling_suggestions.items(), key=lambda x: x[1])
                common_insights.append(f"Most common rolling window: {most_common_rolling[0]} ({most_common_rolling[1]} factors)")
        
        return {
            'total_llm_analyses': len(llm_analyses),
            'factors_with_potential': factors_with_potential,
            'common_insights': common_insights,
            'risk_warnings': list(set(risk_warnings))[:10],  # Unique warnings, top 10
            'sample_analyses': llm_analyses[:5]  # Sample analyses
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
        next_steps = []
        
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
        
        # Next steps
        if deployment_candidates:
            next_steps.append(f"Consider deploying {len(deployment_candidates)} top-performing strategies")
        
        if needs_review:
            next_steps.append(f"Review {len(needs_review)} factors for further optimization")
        
        next_steps.append("Monitor deployed strategies for performance validation")
        next_steps.append("Continue daily analysis to identify new opportunities")
        
        return {
            'deployment_candidates': deployment_candidates,
            'needs_review': needs_review,
            'next_steps': next_steps
        }
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate performance statistics.
        
        :param results: Successful results
        :return: Statistics dictionary
        """
        if not results:
            return {}
        
        sharpe_ratios = []
        calmar_ratios = []
        total_returns = []
        
        for result in results:
            best_result = result.get('best_result', {})
            if best_result:
                sharpe_ratios.append(best_result.get('sharpe_ratio', 0))
                calmar_ratios.append(best_result.get('calmar_ratio', 0))
                total_returns.append(best_result.get('total_return', 0))
        
        if not sharpe_ratios:
            return {}
        
        import statistics as stats
        
        return {
            'sharpe_ratio': {
                'mean': stats.mean(sharpe_ratios),
                'median': stats.median(sharpe_ratios),
                'max': max(sharpe_ratios),
                'min': min(sharpe_ratios),
                'stdev': stats.stdev(sharpe_ratios) if len(sharpe_ratios) > 1 else 0
            },
            'calmar_ratio': {
                'mean': stats.mean(calmar_ratios),
                'median': stats.median(calmar_ratios),
                'max': max(calmar_ratios),
                'min': min(calmar_ratios),
                'stdev': stats.stdev(calmar_ratios) if len(calmar_ratios) > 1 else 0
            },
            'total_return': {
                'mean': stats.mean(total_returns),
                'median': stats.median(total_returns),
                'max': max(total_returns),
                'min': min(total_returns)
            }
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
    
    def save_text_report(self, report: Dict, date_str: str):
        """
        Save comprehensive text report with all data matching JSON format.
        
        :param report: Report dictionary
        :param date_str: Date string
        """
        text_file = self.daily_reports_dir / f"report_{date_str}.txt"
        
        def format_value(value, indent=0):
            """Format a value for text output."""
            prefix = "  " * indent
            if isinstance(value, dict):
                lines = []
                for k, v in value.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{prefix}{k}:")
                        lines.append(format_value(v, indent + 1))
                    else:
                        lines.append(f"{prefix}{k}: {v}")
                return "\n".join(lines)
            elif isinstance(value, list):
                lines = []
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}[{i}]:")
                        lines.append(format_value(item, indent + 1))
                    else:
                        lines.append(f"{prefix}[{i}]: {item}")
                return "\n".join(lines)
            else:
                return f"{prefix}{value}"
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DAILY BACKTEST REPORT - {date_str}\n")
            f.write(f"Generated: {report.get('timestamp', 'N/A')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary (complete)
            summary = report.get('summary', {})
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Statistics (complete)
            stats = report.get('statistics', {})
            if stats:
                f.write("PERFORMANCE STATISTICS\n")
                f.write("-" * 80 + "\n")
                for metric_name, metric_data in stats.items():
                    f.write(f"  {metric_name}:\n")
                    for stat_name, stat_value in metric_data.items():
                        if isinstance(stat_value, float):
                            f.write(f"    {stat_name}: {stat_value:.6f}\n")
                        else:
                            f.write(f"    {stat_name}: {stat_value}\n")
                f.write("\n")
            
            # Top Factors (complete with all parameters)
            top_factors = report.get('top_factors', [])
            if top_factors:
                f.write("TOP FACTORS\n")
                f.write("-" * 80 + "\n")
                for i, factor in enumerate(top_factors, 1):
                    f.write(f"\nFactor #{i}: {factor.get('factor_id', 'N/A')}\n")
                    f.write("-" * 80 + "\n")
                    
                    # Basic info
                    f.write(f"  Asset: {factor.get('asset', 'N/A')}\n")
                    f.write(f"  Factor Name: {factor.get('factor_name', 'N/A')}\n")
                    f.write(f"  Source: {factor.get('source', 'unknown')}\n\n")
                    
                    # Performance metrics
                    f.write("  Performance Metrics:\n")
                    f.write(f"    Sharpe Ratio: {factor.get('sharpe_ratio', 0):.6f}\n")
                    f.write(f"    Calmar Ratio: {factor.get('calmar_ratio', 0):.6f}\n")
                    f.write(f"    Total Return: {factor.get('total_return', 0):.6f} ({factor.get('total_return', 0):.2%})\n")
                    f.write(f"    Annual Return: {factor.get('annual_return', 0):.6f} ({factor.get('annual_return', 0):.2%})\n")
                    f.write(f"    Max Drawdown: {factor.get('max_drawdown', 0):.6f} ({factor.get('max_drawdown', 0):.2%})\n")
                    f.write(f"    Win Rate: {factor.get('win_rate', 0):.6f} ({factor.get('win_rate', 0):.2%})\n")
                    f.write(f"    Number of Trades: {factor.get('num_trades', 0)}\n\n")
                    
                    # Parameters (complete)
                    parameters = factor.get('parameters', {})
                    if parameters:
                        f.write("  Tested Parameters:\n")
                        for param_name, param_value in parameters.items():
                            if isinstance(param_value, list):
                                f.write(f"    {param_name}: {param_value}\n")
                            elif isinstance(param_value, float):
                                f.write(f"    {param_name}: {param_value:.6f}\n")
                            else:
                                f.write(f"    {param_name}: {param_value}\n")
                    f.write("\n")
            
            # LLM Insights (complete)
            llm_insights = report.get('llm_insights', {})
            if llm_insights:
                f.write("LLM INSIGHTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Total LLM Analyses: {llm_insights.get('total_llm_analyses', 0)}\n")
                f.write(f"  Factors with Potential: {llm_insights.get('factors_with_potential', 0)}\n\n")
                
                # Common insights
                common_insights = llm_insights.get('common_insights', [])
                if common_insights:
                    f.write("  Common Insights:\n")
                    for insight in common_insights:
                        f.write(f"    - {insight}\n")
                    f.write("\n")
                
                # Risk warnings
                risk_warnings = llm_insights.get('risk_warnings', [])
                if risk_warnings:
                    f.write("  Risk Warnings:\n")
                    for warning in risk_warnings:
                        f.write(f"    - {warning}\n")
                    f.write("\n")
                
                # Sample analyses (complete)
                sample_analyses = llm_insights.get('sample_analyses', [])
                if sample_analyses:
                    f.write("  Sample LLM Analyses:\n")
                    f.write("-" * 80 + "\n")
                    for i, analysis in enumerate(sample_analyses, 1):
                        f.write(f"\n  Analysis #{i}:\n")
                        f.write(f"    Factor: {analysis.get('factor_name', 'N/A')}\n")
                        f.write(f"    Has Potential: {analysis.get('has_potential', False)}\n")
                        f.write(f"    Confidence: {analysis.get('confidence', 0):.4f}\n")
                        f.write(f"    Reason: {analysis.get('reason', 'N/A')}\n")
                        f.write(f"    Model: {analysis.get('model', 'N/A')}\n")
                        f.write(f"    Timestamp: {analysis.get('timestamp', 'N/A')}\n")
                        
                        # Buy condition suggestion
                        buy_suggestion = analysis.get('buy_condition_suggestion')
                        if buy_suggestion:
                            f.write(f"    Buy Condition Suggestion:\n")
                            f.write(f"      Method: {buy_suggestion.get('method', 'N/A')}\n")
                            f.write(f"      Threshold: {buy_suggestion.get('threshold', 'N/A')}\n")
                            f.write(f"      Reason: {buy_suggestion.get('reason', 'N/A')}\n")
                        
                        # Sell condition suggestion
                        sell_suggestion = analysis.get('sell_condition_suggestion')
                        if sell_suggestion:
                            f.write(f"    Sell Condition Suggestion:\n")
                            f.write(f"      Method: {sell_suggestion.get('method', 'N/A')}\n")
                            f.write(f"      Threshold: {sell_suggestion.get('threshold', 'N/A')}\n")
                            f.write(f"      Reason: {sell_suggestion.get('reason', 'N/A')}\n")
                        
                        # Rolling window suggestion
                        rolling_suggestion = analysis.get('rolling_window_suggestion')
                        if rolling_suggestion:
                            f.write(f"    Rolling Window Suggestion:\n")
                            f.write(f"      Range: {rolling_suggestion.get('range', 'N/A')}\n")
                            f.write(f"      Recommended: {rolling_suggestion.get('recommended', 'N/A')}\n")
                            f.write(f"      Reason: {rolling_suggestion.get('reason', 'N/A')}\n")
                        
                        # Optimization priority
                        opt_priority = analysis.get('optimization_priority')
                        if opt_priority:
                            f.write(f"    Optimization Priority: {opt_priority}\n")
                        
                        # Risk warning
                        risk_warning = analysis.get('risk_warning')
                        if risk_warning:
                            f.write(f"    Risk Warning: {risk_warning}\n")
                        
                        f.write("\n")
            
            # Recommendations (complete)
            recommendations = report.get('recommendations', {})
            if recommendations:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 80 + "\n")
                
                deployment = recommendations.get('deployment_candidates', [])
                if deployment:
                    f.write(f"  Deployment Candidates ({len(deployment)}):\n")
                    for candidate in deployment:
                        f.write(f"    Factor ID: {candidate.get('factor_id', 'N/A')}\n")
                        f.write(f"      Sharpe Ratio: {candidate.get('sharpe_ratio', 0):.6f}\n")
                        f.write(f"      Calmar Ratio: {candidate.get('calmar_ratio', 0):.6f}\n")
                        f.write(f"      Reason: {candidate.get('reason', 'N/A')}\n")
                    f.write("\n")
                
                needs_review = recommendations.get('needs_review', [])
                if needs_review:
                    f.write(f"  Needs Review ({len(needs_review)}):\n")
                    for item in needs_review:
                        f.write(f"    Factor ID: {item.get('factor_id', 'N/A')}\n")
                        f.write(f"      Sharpe Ratio: {item.get('sharpe_ratio', 0):.6f}\n")
                        f.write(f"      Reason: {item.get('reason', 'N/A')}\n")
                    f.write("\n")
                
                next_steps = recommendations.get('next_steps', [])
                if next_steps:
                    f.write("  Next Steps:\n")
                    for step in next_steps:
                        f.write(f"    - {step}\n")
                    f.write("\n")
            
            # Performance metrics (complete)
            performance = report.get('performance', {})
            if performance:
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 80 + "\n")
                if isinstance(performance, dict):
                    if 'session_duration_seconds' in performance:
                        f.write(f"  Session Duration: {performance.get('session_duration_seconds', 0):.2f} seconds\n")
                    if 'total_operations' in performance:
                        f.write(f"  Total Operations: {performance.get('total_operations', 0)}\n")
                    if 'operation_stats' in performance:
                        f.write("  Operation Statistics:\n")
                        for op_name, op_stats in performance.get('operation_stats', {}).items():
                            f.write(f"    {op_name}:\n")
                            for stat_name, stat_value in op_stats.items():
                                if isinstance(stat_value, float):
                                    f.write(f"      {stat_name}: {stat_value:.6f}\n")
                                else:
                                    f.write(f"      {stat_name}: {stat_value}\n")
                f.write("\n")
            
            # All results count
            all_results_count = report.get('all_results_count', 0)
            f.write("METADATA\n")
            f.write("-" * 80 + "\n")
            f.write(f"  All Results Count: {all_results_count}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logging.info(f"Text report saved to {text_file}")
    
    def save_llm_analysis_report(self, report: Dict, date_str: str):
        """
        Save LLM analysis report separately.
        
        :param report: Report dictionary
        :param date_str: Date string
        """
        llm_insights = report.get('llm_insights', {})
        if not llm_insights:
            return
        
        analysis_file = self.llm_analysis_dir / f"analysis_{date_str}.txt"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"LLM ANALYSIS REPORT - {date_str}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total LLM Analyses: {llm_insights.get('total_llm_analyses', 0)}\n")
            f.write(f"Factors with Potential: {llm_insights.get('factors_with_potential', 0)}\n\n")
            
            # Common insights
            common_insights = llm_insights.get('common_insights', [])
            if common_insights:
                f.write("COMMON INSIGHTS\n")
                f.write("-" * 80 + "\n")
                for insight in common_insights:
                    f.write(f"{insight}\n")
                f.write("\n")
            
            # Risk warnings
            risk_warnings = llm_insights.get('risk_warnings', [])
            if risk_warnings:
                f.write("RISK WARNINGS\n")
                f.write("-" * 80 + "\n")
                for warning in risk_warnings:
                    f.write(f"{warning}\n")
                f.write("\n")
            
            # Sample analyses
            sample_analyses = llm_insights.get('sample_analyses', [])
            if sample_analyses:
                f.write("SAMPLE LLM ANALYSES\n")
                f.write("-" * 80 + "\n")
                for i, analysis in enumerate(sample_analyses, 1):
                    f.write(f"\nAnalysis {i}:\n")
                    f.write(f"Has Potential: {analysis.get('has_potential', False)}\n")
                    f.write(f"Confidence: {analysis.get('confidence', 0):.2f}\n")
                    f.write(f"Reason: {analysis.get('reason', 'N/A')}\n")
                    if analysis.get('optimization_priority'):
                        f.write(f"Optimization Priority: {analysis.get('optimization_priority')}\n")
                    f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        logging.info(f"LLM analysis report saved to {analysis_file}")
    
    def save_all_reports(self, report: Dict, date_str: str):
        """
        Save all report formats.
        
        :param report: Report dictionary
        :param date_str: Date string
        """
        self.save_json_report(report, date_str)
        self.save_text_report(report, date_str)
        self.save_llm_analysis_report(report, date_str)
        
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
