"""
factor_screening.py

Two-stage factor screening system:
- Stage 1: Complete grid search backtest
- Decision: Based on best Sharpe and Calmar ratios
- Stage 2: LLM-guided optimization (only for promising factors)

Features:
- Complete parameter grid search
- Sharpe and Calmar based filtering
- LLM-guided optimization
- Performance tracking
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from itertools import product
from tqdm import tqdm

from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
from core.llm_client import LLMClient
from core.performance_monitor import PerformanceMonitor
from core.factor_status_tracker import get_status_tracker, FactorStatus
from utils.log_config import should_log

# Configure logging (minimal - use progress bars instead)
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors by default
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TwoStageFactorScreening:
    """
    Two-stage factor screening system.
    
    Stage 1: Complete grid search to find best parameters
    Stage 2: LLM-guided optimization for promising factors
    """
    
    def __init__(self,
                 strategy: EnhancedNonPriceStrategy,
                 llm_client: LLMClient = None,
                 min_sharpe_threshold: float = 0.5,
                 min_calmar_threshold: float = 0.3,
                 performance_monitor: PerformanceMonitor = None):
        """
        Initialize two-stage screening system.
        
        :param strategy: EnhancedNonPriceStrategy instance
        :param llm_client: LLM client for stage 2 optimization
        :param min_sharpe_threshold: Minimum Sharpe ratio to continue
        :param min_calmar_threshold: Minimum Calmar ratio to continue
        :param performance_monitor: Performance monitor instance
        """
        self.strategy = strategy
        self.llm_client = llm_client
        self.min_sharpe_threshold = min_sharpe_threshold
        self.min_calmar_threshold = min_calmar_threshold
        self.performance_monitor = performance_monitor
        self.status_tracker = get_status_tracker()
        
        logging.info("Two-Stage Factor Screening initialized")
        logging.info(f"  Min Sharpe threshold: {min_sharpe_threshold}")
        logging.info(f"  Min Calmar threshold: {min_calmar_threshold}")
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio.
        
        :param annual_return: Annual return (as decimal, e.g., 0.2 for 20%)
        :param max_drawdown: Maximum drawdown (as decimal, e.g., -0.15 for -15%)
        :return: Calmar ratio
        """
        if max_drawdown == 0 or abs(max_drawdown) < 0.001:
            return 0.0
        
        # Calmar = Annual Return / |Max Drawdown|
        calmar = annual_return / abs(max_drawdown)
        return calmar
    
    def stage1_grid_search(self,
                          asset: str,
                          factor_name: str,
                          param_grid: Dict,
                          date_range: Tuple[str, str] = None) -> Dict:
        """
        Stage 1: Complete grid search backtest.
        
        Tests all parameter combinations and finds the best result.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param factor_name: Factor name
        :param param_grid: Parameter grid dictionary
            Example: {
                'rolling': [1, 5, 10, 20],
                'long_param': [-0.1, -0.05, -0.04],
                'short_param': [0.05, 0.1, 0.14]
            }
        :param date_range: Date range tuple (start, end) or None
        :return: Dictionary with grid search results
        """
        logging.info(f"Stage 1: Starting grid search for {asset}/{factor_name}")
        
        # Update status: Stage 1 running
        factor_id = f"{asset}_{factor_name}"
        self.status_tracker.register_factor(factor_id, factor_name, asset)
        self.status_tracker.update_status(factor_id, FactorStatus.STAGE1_RUNNING)
        
        if self.performance_monitor:
            self.performance_monitor.start_operation("stage1_grid_search")
        
        # Extract param_method (should be a single value, not part of grid)
        param_method = None
        if 'param_method' in param_grid:
            param_method_list = param_grid.get('param_method', [])
            if param_method_list and len(param_method_list) > 0:
                param_method = param_method_list[0]
        
        # Generate all parameter combinations (excluding param_method)
        grid_for_combinations = {k: v for k, v in param_grid.items() if k != 'param_method'}
        param_names = list(grid_for_combinations.keys())
        param_values = list(grid_for_combinations.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        logging.info(f"  Total parameter combinations: {total_combinations}")
        if param_method:
            logging.info(f"  Using param_method: {param_method}")
        
        results = []
        best_result = None
        best_sharpe = -np.inf
        
        # Use progress bar instead of frequent logging
        progress_bar = tqdm(
            enumerate(param_combinations, 1),
            total=total_combinations,
            desc=f"Grid search {asset}/{factor_name[:30]}",
            unit="comb",
            disable=not should_log('info')  # Disable if quiet mode
        )
        
        # Test each parameter combination
        for i, combination in progress_bar:
            params = dict(zip(param_names, combination))
            
            # Add required parameters
            params['initial_capital'] = self.strategy.initial_capital
            params['lot_size'] = self.strategy.min_lot_size
            if date_range:
                params['date_range'] = date_range
            
            # Add param_method if specified
            if param_method:
                params['param_method'] = param_method
            
            try:
                # Run backtest
                result = self.strategy.run_backtest(asset, factor_name, params)
                
                if result:
                    # Calculate Calmar ratio
                    annual_return = result.get('annual_return', 0)
                    max_drawdown = result.get('max_drawdown', 0)
                    calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)
                    
                    # Add metrics
                    result['calmar_ratio'] = calmar_ratio
                    result['params'] = params
                    result['combination_index'] = i
                    
                    results.append(result)
                    
                    # Track best result
                    sharpe_ratio = result.get('sharpe_ratio', 0)
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_result = result
                    
                    # Update progress bar with best Sharpe
                    if should_log('info'):
                        progress_bar.set_postfix({'Best Sharpe': f'{best_sharpe:.4f}'})
            
            except Exception as e:
                if should_log('error'):
                    logging.error(f"  Error testing combination {i}: {e}")
                continue
        
        # Close progress bar
        progress_bar.close()
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("stage1_grid_search")
        
        if not results:
            logging.warning(f"  No valid results from grid search")
            # Update status: Stage 1 failed
            self.status_tracker.update_status(
                factor_id,
                FactorStatus.STAGE1_FAILED,
                error_message='No valid results from grid search'
            )
            return {
                'stage': 'stage1',
                'status': 'failed',
                'reason': 'No valid results',
                'total_combinations': total_combinations,
                'results': []
            }
        
        # Summary statistics
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        calmar_ratios = [r.get('calmar_ratio', 0) for r in results]
        best_calmar = best_result.get('calmar_ratio', 0) if best_result else 0
        
        summary = {
            'stage': 'stage1',
            'status': 'completed',
            'total_combinations': total_combinations,
            'valid_results': len(results),
            'best_result': best_result,
            'best_sharpe': best_sharpe,
            'best_calmar': best_calmar,
            'avg_sharpe': float(np.mean(sharpe_ratios)) if sharpe_ratios else 0,
            'max_sharpe': float(np.max(sharpe_ratios)) if sharpe_ratios else 0,
            'min_sharpe': float(np.min(sharpe_ratios)) if sharpe_ratios else 0,
            'avg_calmar': float(np.mean(calmar_ratios)) if calmar_ratios else 0,
            'max_calmar': float(np.max(calmar_ratios)) if calmar_ratios else 0,
            'min_calmar': float(np.min(calmar_ratios)) if calmar_ratios else 0,
            'results': results
        }
        
        if should_log('info'):
            logging.info(f"Stage 1 completed: Best Sharpe {best_sharpe:.4f}, Best Calmar {best_calmar:.4f}")
        
        # Update status: Stage 1 completed
        self.status_tracker.update_status(
            factor_id,
            FactorStatus.STAGE1_COMPLETED,
            stage_result=summary
        )
        
        return summary
    
    def should_continue_to_stage2(self, stage1_result: Dict) -> Tuple[bool, str]:
        """
        Decide whether to continue to Stage 2 based on Stage 1 results.
        
        Decision criteria:
        - Best Sharpe ratio >= min_sharpe_threshold
        - Best Calmar ratio >= min_calmar_threshold
        
        :param stage1_result: Stage 1 grid search results
        :return: Tuple (should_continue, reason)
        """
        if stage1_result.get('status') != 'completed':
            return False, f"Stage 1 failed: {stage1_result.get('reason', 'Unknown')}"
        
        best_sharpe = stage1_result.get('best_sharpe', 0)
        best_calmar = stage1_result.get('best_calmar', 0)
        
        # Check Sharpe threshold
        if best_sharpe < self.min_sharpe_threshold:
            return False, f"Best Sharpe {best_sharpe:.4f} < threshold {self.min_sharpe_threshold}"
        
        # Check Calmar threshold
        if best_calmar < self.min_calmar_threshold:
            return False, f"Best Calmar {best_calmar:.4f} < threshold {self.min_calmar_threshold}"
        
        return True, f"Passed thresholds (Sharpe: {best_sharpe:.4f}, Calmar: {best_calmar:.4f})"
    
    def stage2_llm_optimization(self,
                                asset: str,
                                factor_name: str,
                                stage1_best_result: Dict,
                                historical_context: str = "",
                                factor_data: pd.DataFrame = None,
                                stage1_results: List[Dict] = None,
                                max_iterations: int = 5,
                                min_improvement: float = 0.01) -> Dict:
        """
        Stage 2: LLM-guided iterative optimization with full data.
        
        Uses Stage 1 best result as starting point and LLM to iteratively optimize.
        Sends complete factor data to LLM for better analysis.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param stage1_best_result: Best result from Stage 1
        :param historical_context: Historical context string for LLM
        :param factor_data: Complete factor data DataFrame (for full data analysis)
        :param stage1_results: All Stage 1 results (top N for context)
        :param max_iterations: Maximum optimization iterations
        :param min_improvement: Minimum improvement to continue iteration
        :return: LLM optimization results
        """
        if not self.llm_client:
            logging.warning("LLM client not available, skipping Stage 2")
            # Update status: Stage 2 skipped
            factor_id = f"{asset}_{factor_name}"
            self.status_tracker.update_status(
                factor_id,
                FactorStatus.STAGE2_FAILED,
                error_message='LLM client not available'
            )
            return {
                'stage': 'stage2',
                'status': 'skipped',
                'reason': 'LLM client not available'
            }
        
        logging.info(f"Stage 2: Starting LLM iterative optimization for {asset}/{factor_name}")
        logging.info(f"  Max iterations: {max_iterations}")
        logging.info(f"  Min improvement threshold: {min_improvement}")
        
        # Update status: Stage 2 running
        factor_id = f"{asset}_{factor_name}"
        self.status_tracker.update_status(factor_id, FactorStatus.STAGE2_RUNNING)
        
        if self.performance_monitor:
            self.performance_monitor.start_operation("stage2_llm_optimization")
        
        # Prepare initial parameters from Stage 1 best result
        current_params = stage1_best_result.get('params', {}).copy()
        current_best_sharpe = stage1_best_result.get('sharpe_ratio', 0)
        current_best_result = stage1_best_result.copy()
        
        # Store iteration history
        iteration_results = []
        all_llm_analyses = []
        
        # Get top results from Stage 1 for context (if available)
        top_stage1_results = []
        if stage1_results:
            # Get top 5 results from Stage 1
            sorted_results = sorted(
                stage1_results,
                key=lambda x: x.get('sharpe_ratio', 0),
                reverse=True
            )[:5]
            top_stage1_results = sorted_results
        
        # Iterative optimization loop
        for iteration in range(1, max_iterations + 1):
            logging.info(f"  Iteration {iteration}/{max_iterations}")
            
            try:
                # Prepare data for LLM (with full factor data if available)
                llm_input = {
                    'factor_name': f"{asset}_{factor_name}",
                    'current_params': current_params,
                    'current_result': {
                        'sharpe_ratio': current_best_result.get('sharpe_ratio', 0),
                        'calmar_ratio': current_best_result.get('calmar_ratio', 0),
                        'total_return': current_best_result.get('total_return', 0),
                        'annual_return': current_best_result.get('annual_return', 0),
                        'max_drawdown': current_best_result.get('max_drawdown', 0),
                        'win_rate': current_best_result.get('win_rate', 0),
                        'num_trades': current_best_result.get('num_trades', 0)
                    },
                    'iteration': iteration,
                    'previous_iterations': iteration_results[-2:] if iteration_results else [],  # Last 2 iterations
                    'top_stage1_results': top_stage1_results,  # Top 5 from Stage 1
                    'historical_context': historical_context,
                    'factor_data': None  # Will be set if factor_data provided
                }
                
                # Add complete factor data if available
                if factor_data is not None and len(factor_data) > 0:
                    # Convert DataFrame to list of dicts for JSON serialization
                    # Send full data to LLM for complete analysis
                    llm_input['factor_data'] = factor_data.to_dict('records')
                    llm_input['data_summary'] = {
                        'total_rows': len(factor_data),
                        'date_range': {
                            'start': str(factor_data.iloc[0]['timestamp']) if 'timestamp' in factor_data.columns else None,
                            'end': str(factor_data.iloc[-1]['timestamp']) if 'timestamp' in factor_data.columns else None
                        }
                    }
                    logging.info(f"  Sending full factor data: {len(factor_data)} rows")
                
                # Request LLM analysis with full context
                llm_analysis = self.llm_client.analyze_factor_iterative(llm_input)
                
                if not llm_analysis:
                    logging.warning(f"  LLM analysis failed at iteration {iteration}")
                    # Update status: Stage 2 failed
                    factor_id = f"{asset}_{factor_name}"
                    self.status_tracker.update_status(
                        factor_id,
                        FactorStatus.STAGE2_FAILED,
                        error_message=f'LLM analysis failed at iteration {iteration}'
                    )
                    break
                
                all_llm_analyses.append(llm_analysis)
                
                # Extract optimization suggestions
                optimization_suggestions = self._extract_optimization_suggestions(llm_analysis)
                
                if not optimization_suggestions:
                    logging.warning(f"  No optimization suggestions at iteration {iteration}")
                    # Update status: Stage 2 failed (no suggestions)
                    factor_id = f"{asset}_{factor_name}"
                    self.status_tracker.update_status(
                        factor_id,
                        FactorStatus.STAGE2_FAILED,
                        error_message=f'No optimization suggestions at iteration {iteration}'
                    )
                    break
                
                # Test top suggestion (most promising)
                best_suggestion_result = None
                best_suggestion_sharpe = current_best_sharpe
                
                # Test top 2 suggestions
                for suggestion in optimization_suggestions[:2]:
                    try:
                        # Create new parameters based on suggestion
                        new_params = self._apply_suggestion(current_params.copy(), suggestion)
                        
                        # Validate parameters
                        if not self._validate_params(new_params):
                            continue
                        
                        # Add required parameters
                        new_params['initial_capital'] = self.strategy.initial_capital
                        new_params['lot_size'] = self.strategy.min_lot_size
                        if 'date_range' in current_params:
                            new_params['date_range'] = current_params['date_range']
                        if 'param_method' in current_params:
                            new_params['param_method'] = current_params['param_method']
                        
                        # Run backtest
                        result = self.strategy.run_backtest(asset, factor_name, new_params)
                        
                        if result:
                            annual_return = result.get('annual_return', 0)
                            max_drawdown = result.get('max_drawdown', 0)
                            result['calmar_ratio'] = self._calculate_calmar_ratio(annual_return, max_drawdown)
                            result['params'] = new_params
                            result['suggestion'] = suggestion
                            result['iteration'] = iteration
                            
                            sharpe = result.get('sharpe_ratio', 0)
                            if sharpe > best_suggestion_sharpe:
                                best_suggestion_sharpe = sharpe
                                best_suggestion_result = result
                    
                    except Exception as e:
                        logging.error(f"  Error testing suggestion at iteration {iteration}: {e}")
                        continue
                
                # Check if we found improvement
                if best_suggestion_result and best_suggestion_sharpe > current_best_sharpe:
                    improvement = best_suggestion_sharpe - current_best_sharpe
                    logging.info(
                        f"  Iteration {iteration}: Improvement {improvement:.4f} "
                        f"(Sharpe: {current_best_sharpe:.4f} -> {best_suggestion_sharpe:.4f})"
                    )
                    
                    # Update current best
                    current_params = best_suggestion_result['params'].copy()
                    current_best_sharpe = best_suggestion_sharpe
                    current_best_result = best_suggestion_result
                    
                    iteration_results.append({
                        'iteration': iteration,
                        'params': current_params.copy(),
                        'result': current_best_result.copy(),
                        'improvement': improvement,
                        'llm_analysis': llm_analysis
                    })
                    
                    # Check if improvement is significant enough to continue
                    if improvement < min_improvement:
                        logging.info(f"  Improvement {improvement:.4f} < threshold {min_improvement}, stopping")
                        break
                else:
                    logging.info(f"  Iteration {iteration}: No improvement found")
                    # If no improvement for 2 consecutive iterations, stop
                    if len(iteration_results) > 0 and iteration > 2:
                        no_improvement_count = iteration - len(iteration_results)
                        if no_improvement_count >= 2:
                            logging.info("  No improvement for 2 consecutive iterations, stopping")
                            break
            
            except Exception as e:
                logging.error(f"  Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("stage2_llm_optimization")
        
        # Prepare final result
        total_improvement = current_best_sharpe - stage1_best_result.get('sharpe_ratio', 0)
        
        logging.info(f"Stage 2 completed: {len(iteration_results)} successful iterations")
        logging.info(f"  Final Sharpe: {current_best_sharpe:.4f} (improvement: {total_improvement:.4f})")
        
        result = {
            'stage': 'stage2',
            'status': 'completed',
            'iterations': len(iteration_results),
            'max_iterations': max_iterations,
            'llm_analyses': all_llm_analyses,
            'iteration_results': iteration_results,
            'best_optimized': current_best_result if iteration_results else None,
            'improvement': total_improvement,
            'initial_sharpe': stage1_best_result.get('sharpe_ratio', 0),
            'final_sharpe': current_best_sharpe
        }
        
        # Update status: Stage 2 completed
        factor_id = f"{asset}_{factor_name}"
        self.status_tracker.update_status(
            factor_id,
            FactorStatus.STAGE2_COMPLETED,
            stage_result=result
        )
        
        return result
    
    def _validate_params(self, params: Dict) -> bool:
        """
        Validate parameter values are within reasonable ranges.
        Simplified validation - parameters are already generated from data range.
        
        :param params: Parameter dictionary
        :return: True if valid, False otherwise
        """
        rolling = params.get('rolling')
        if rolling is not None and (not isinstance(rolling, (int, float)) or rolling < 1 or rolling > 100):
            return False
        
        long_param = params.get('long_param')
        if long_param is not None and (not isinstance(long_param, (int, float)) or long_param > 0.5 or long_param < -1.0):
            return False
        
        short_param = params.get('short_param')
        if short_param is not None and (not isinstance(short_param, (int, float)) or short_param < -0.5 or short_param > 1.0):
            return False
        
        return True
    
    def _extract_optimization_suggestions(self, llm_analysis: Dict) -> List[Dict]:
        """
        Extract optimization suggestions from LLM analysis.
        
        :param llm_analysis: LLM analysis result
        :return: List of optimization suggestions
        """
        suggestions = []
        
        # Extract rolling window suggestion
        rolling_suggestion = llm_analysis.get('rolling_window_suggestion', {})
        if rolling_suggestion:
            suggestions.append({
                'type': 'rolling',
                'range': rolling_suggestion.get('range', [1, 20]),
                'recommended': rolling_suggestion.get('recommended', 10),
                'reason': rolling_suggestion.get('reason', '')
            })
        
        # Extract buy condition suggestion
        buy_suggestion = llm_analysis.get('buy_condition_suggestion', {})
        if buy_suggestion:
            suggestions.append({
                'type': 'long_param',
                'method': buy_suggestion.get('method', 'percentage'),
                'threshold': buy_suggestion.get('threshold', 0),
                'reason': buy_suggestion.get('reason', '')
            })
        
        # Extract sell condition suggestion
        sell_suggestion = llm_analysis.get('sell_condition_suggestion', {})
        if sell_suggestion:
            suggestions.append({
                'type': 'short_param',
                'method': sell_suggestion.get('method', 'percentage'),
                'threshold': sell_suggestion.get('threshold', 0),
                'reason': sell_suggestion.get('reason', '')
            })
        
        return suggestions
    
    def _apply_suggestion(self, current_params: Dict, suggestion: Dict) -> Dict:
        """
        Apply LLM suggestion to current parameters.
        
        :param current_params: Current parameter dictionary
        :param suggestion: LLM suggestion dictionary
        :return: New parameter dictionary
        """
        new_params = current_params.copy()
        
        suggestion_type = suggestion.get('type')
        
        if suggestion_type == 'rolling':
            new_params['rolling'] = suggestion.get('recommended', current_params.get('rolling', 10))
        
        elif suggestion_type == 'long_param':
            new_params['long_param'] = suggestion.get('threshold', current_params.get('long_param', -0.04))
        
        elif suggestion_type == 'short_param':
            new_params['short_param'] = suggestion.get('threshold', current_params.get('short_param', 0.14))
        
        return new_params
    
    def run_complete_screening(self,
                              asset: str,
                              factor_name: str,
                              param_grid: Dict,
                              date_range: Tuple[str, str] = None,
                              historical_context: str = "",
                              factor_data: pd.DataFrame = None) -> Dict:
        """
        Run complete two-stage screening process.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param param_grid: Parameter grid for Stage 1
        :param date_range: Date range tuple or None
        :param historical_context: Historical context for LLM
        :param factor_data: Complete factor data DataFrame (for Stage 2 iterative optimization)
        :return: Complete screening results
        """
        logging.info(f"Starting complete screening for {asset}/{factor_name}")
        
        factor_id = f"{asset}_{factor_name}"
        
        try:
            # Register factor if not exists
            self.status_tracker.register_factor(factor_id, factor_name, asset)
            
            # Stage 1: Grid search (status updated inside stage1_grid_search)
            stage1_result = self.stage1_grid_search(asset, factor_name, param_grid, date_range)
            
            # Check if Stage 1 failed
            if stage1_result.get('status') == 'failed':
                result = {
                    'asset': asset,
                    'factor_name': factor_name,
                    'timestamp': datetime.now().isoformat(),
                    'stage1': stage1_result,
                    'stage2': None,
                    'final_decision': stage1_result.get('reason', 'Stage 1 failed'),
                    'best_result': None,
                    'best_source': None
                }
                return result
            
            # Decision: Continue to Stage 2?
            should_continue, reason = self.should_continue_to_stage2(stage1_result)
            
            result = {
                'asset': asset,
                'factor_name': factor_name,
                'timestamp': datetime.now().isoformat(),
                'stage1': stage1_result,
                'stage2': None,
                'final_decision': reason
            }
            
            if should_continue:
                logging.info(f"Continuing to Stage 2: {reason}")
                
                # Get all Stage 1 results for context
                stage1_results = stage1_result.get('results', [])
                
                # Stage 2: LLM iterative optimization with full data (status updated inside)
                stage2_result = self.stage2_llm_optimization(
                    asset,
                    factor_name,
                    stage1_result['best_result'],
                    historical_context,
                    factor_data=factor_data,
                    stage1_results=stage1_results,
                    max_iterations=5,
                    min_improvement=0.01
                )
                
                result['stage2'] = stage2_result
                
                # Determine final best result
                if stage2_result.get('best_optimized'):
                    stage2_sharpe = stage2_result['best_optimized'].get('sharpe_ratio', 0)
                    stage1_sharpe = stage1_result['best_result'].get('sharpe_ratio', 0)
                    
                    if stage2_sharpe > stage1_sharpe:
                        result['best_result'] = stage2_result['best_optimized']
                        result['best_source'] = 'stage2_llm_iterative'
                    else:
                        result['best_result'] = stage1_result['best_result']
                        result['best_source'] = 'stage1_grid_search'
                else:
                    result['best_result'] = stage1_result['best_result']
                    result['best_source'] = 'stage1_grid_search'
                
                # Update status: Completed
                if stage2_result.get('status') == 'completed':
                    self.status_tracker.update_status(factor_id, FactorStatus.COMPLETED)
                elif stage2_result.get('status') == 'failed':
                    self.status_tracker.update_status(
                        factor_id,
                        FactorStatus.STAGE2_FAILED,
                        error_message=stage2_result.get('reason', 'Unknown')
                    )
            else:
                logging.info(f"Skipping Stage 2: {reason}")
                result['best_result'] = stage1_result.get('best_result')
                result['best_source'] = 'stage1_grid_search'
                
                # Update status: Skipped (Stage 2 not reached)
                self.status_tracker.update_status(factor_id, FactorStatus.SKIPPED)
            
            logging.info(f"Screening completed for {asset}/{factor_name}")
            logging.info(f"  Best Sharpe: {result['best_result'].get('sharpe_ratio', 0):.4f}")
            logging.info(f"  Best Calmar: {result['best_result'].get('calmar_ratio', 0):.4f}")
            
            return result
        
        except Exception as e:
            # Update status: Failed
            logging.error(f"Error in complete screening for {asset}/{factor_name}: {e}")
            self.status_tracker.update_status(
                factor_id,
                FactorStatus.FAILED,
                error_message=str(e)
            )
            raise


if __name__ == "__main__":
    # Test the screening system
    from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
    from core.llm_client import LLMClient
    
    print("Testing Two-Stage Factor Screening...")
    
    # Initialize components
    strategy = EnhancedNonPriceStrategy(
        initial_capital=10000,
        min_lot_size=0.001,
        use_api=True,
        param_method='pct_change'
    )
    
    # LLM client (optional for testing)
    llm_client = LLMClient(api_url="http://localhost:11434/api/chat")
    
    # Create screening system
    screening = TwoStageFactorScreening(
        strategy=strategy,
        llm_client=llm_client,
        min_sharpe_threshold=0.5,
        min_calmar_threshold=0.3
    )
    
    # Test parameter grid
    param_grid = {
        'rolling': [1, 5, 10],
        'long_param': [-0.04],
        'short_param': [0.14]
    }
    
    # Run screening (commented out for testing)
    # result = screening.run_complete_screening(
    #     asset='BTC',
    #     factor_name='indicators/sopr_account_based',
    #     param_grid=param_grid,
    #     date_range=('2023-01-01', '2025-08-10')
    # )
    # 
    # print("\nScreening Result:")
    # print(f"  Best Sharpe: {result['best_result'].get('sharpe_ratio', 0):.4f}")
    # print(f"  Best Calmar: {result['best_result'].get('calmar_ratio', 0):.4f}")
    # print(f"  Source: {result['best_source']}")
