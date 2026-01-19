"""
llm_orchestrator.py

LLM Orchestrator that guides LLM through complete workflow.

Features:
- Factor checking and validation
- Factor exploration coordination
- LLM analysis orchestration
- Parameter optimization guidance
- Result logging and context management
"""

import sys
import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
from core.llm_client import LLMClient
from core.context_manager import ContextManager
from core.factor_screening import TwoStageFactorScreening
from core.llm_scheduler import IntelligentLLMScheduler
from core.data_cache import LightweightDataCache
from core.performance_monitor import PerformanceMonitor, get_monitor
from core.intelligent_param_generator import IntelligentParamGenerator
from utils.local_data_loader import LocalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LLMOrchestrator:
    """
    Orchestrator that guides LLM through complete factor analysis workflow.
    
    Workflow:
    1. Factor check (data quality, completeness)
    2. Factor exploration (two-stage screening)
    3. LLM analysis (batch or individual)
    4. Parameter optimization (apply LLM suggestions)
    5. Result logging (save to context manager)
    """
    
    def __init__(self,
                 strategy: EnhancedNonPriceStrategy,
                 llm_client: LLMClient,
                 context_manager: ContextManager,
                 screening: TwoStageFactorScreening,
                 scheduler: IntelligentLLMScheduler,
                 data_cache: LightweightDataCache = None,
                 performance_monitor: PerformanceMonitor = None,
                 param_generator: IntelligentParamGenerator = None):
        """
        Initialize LLM orchestrator.
        
        :param strategy: EnhancedNonPriceStrategy instance
        :param llm_client: LLM client instance
        :param context_manager: Context manager instance
        :param screening: Two-stage screening system
        :param scheduler: Intelligent LLM scheduler
        :param data_cache: Data cache instance (optional)
        :param performance_monitor: Performance monitor instance
        """
        self.strategy = strategy
        self.llm_client = llm_client
        self.context_manager = context_manager
        self.screening = screening
        self.scheduler = scheduler
        self.data_cache = data_cache
        self.performance_monitor = performance_monitor or get_monitor()
        
        self.data_loader = LocalDataLoader()
        self.param_generator = param_generator
        
        logging.info("LLM Orchestrator initialized")
    
    def check_factor(self, asset: str, factor_name: str) -> Dict:
        """
        Step 1: Check factor data quality and completeness.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :return: Check result dictionary with factor_data included
        """
        if self.performance_monitor:
            self.performance_monitor.start_operation("factor_check")
        
        logging.info(f"Checking factor: {asset}/{factor_name}")
        
        check_result = {
            'asset': asset,
            'factor_name': factor_name,
            'factor_id': f"{asset}_{factor_name}",
            'status': 'unknown',
            'data_quality': {},
            'errors': [],
            'factor_data': None  # Store factor data for parameter generation
        }
        
        try:
            # Try to load data
            price_data, factor_data = self.data_loader.load_data_pair(asset, factor_name)
            
            if price_data is None or factor_data is None:
                check_result['status'] = 'failed'
                check_result['errors'].append("Failed to load data")
                return check_result
            
            # Store factor data for parameter generation
            check_result['factor_data'] = factor_data.copy()
            
            # Check data quality
            data_quality = {
                'price_data_points': len(price_data),
                'factor_data_points': len(factor_data),
                'price_missing': price_data['value'].isna().sum(),
                'factor_missing': factor_data['value'].isna().sum(),
                'price_missing_pct': (price_data['value'].isna().sum() / len(price_data)) * 100,
                'factor_missing_pct': (factor_data['value'].isna().sum() / len(factor_data)) * 100
            }
            
            check_result['data_quality'] = data_quality
            
            # Validate data quality
            if data_quality['price_data_points'] < 30:
                check_result['status'] = 'failed'
                check_result['errors'].append(f"Insufficient price data: {data_quality['price_data_points']} points")
            
            if data_quality['factor_data_points'] < 30:
                check_result['status'] = 'failed'
                check_result['errors'].append(f"Insufficient factor data: {data_quality['factor_data_points']} points")
            
            if data_quality['price_missing_pct'] > 20:
                check_result['status'] = 'warning'
                check_result['errors'].append(f"High price data missing: {data_quality['price_missing_pct']:.1f}%")
            
            if data_quality['factor_missing_pct'] > 20:
                check_result['status'] = 'warning'
                check_result['errors'].append(f"High factor data missing: {data_quality['factor_missing_pct']:.1f}%")
            
            if check_result['status'] == 'unknown':
                check_result['status'] = 'passed'
            
            logging.info(f"Factor check completed: {check_result['status']}")
            
        except Exception as e:
            check_result['status'] = 'failed'
            check_result['errors'].append(f"Exception during check: {str(e)}")
            logging.error(f"Error checking factor {asset}/{factor_name}: {e}")
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("factor_check")
        
        return check_result
    
    def explore_factor(self,
                      asset: str,
                      factor_name: str,
                      param_grid: Dict = None,
                      date_range: Tuple[str, str] = None,
                      factor_data: pd.DataFrame = None) -> Dict:
        """
        Step 2: Explore factor using two-stage screening.
        
        If param_grid is None, will use intelligent parameter generation.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param param_grid: Parameter grid for grid search (if None, will generate intelligently)
        :param date_range: Date range tuple or None
        :param factor_data: Factor data DataFrame (for intelligent parameter generation)
        :return: Exploration results
        """
        if self.performance_monitor:
            self.performance_monitor.start_operation("factor_exploration")
        
        logging.info(f"Exploring factor: {asset}/{factor_name}")
        
        # Always try to load factor data first for intelligent parameter generation
        if factor_data is None:
            try:
                _, factor_data = self.data_loader.load_data_pair(asset, factor_name)
                if factor_data is not None:
                    logging.info(f"Loaded factor data for parameter generation: {len(factor_data)} rows")
            except Exception as e:
                logging.warning(f"Could not load factor data for {asset}/{factor_name}: {e}")
                factor_data = None
        
        # Generate parameter grid if not provided
        # Default: Use simple min/max method (fast, reliable)
        # Optional: Use AI method if explicitly requested
        if param_grid is None:
            if self.param_generator and factor_data is not None:
                # Use simple min/max method by default (use_ai=False)
                # Set use_ai=True if you want AI-based generation
                use_ai_generation = False  # Can be made configurable
                
                logging.info(f"Generating parameter space using {'AI' if use_ai_generation else 'min/max'} method...")
                param_grid = self.param_generator.analyze_and_generate_params(
                    asset=asset,
                    factor_name=factor_name,
                    factor_data=factor_data,
                    use_ai=use_ai_generation
                )
            else:
                logging.warning("No parameter generator or factor data, using default grid")
                param_grid = {
                    'rolling': [1, 3, 5, 7, 10, 14, 20, 30, 50],
                    'long_param': [-0.3, -0.25, -0.2, -0.15, -0.1, -0.08, -0.05, -0.03, -0.02, -0.01],
                    'short_param': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3],
                    'param_method': ['pct_change']
                }
                # Total combinations: 9 × 10 × 10 = 900 combinations
        
        # Get historical context
        factor_id = f"{asset}_{factor_name}"
        historical_context = self.context_manager.get_factor_summary(factor_id)
        
        # Load factor data if not provided (for Stage 2 iterative optimization)
        if factor_data is None:
            try:
                _, factor_data = self.data_loader.load_data_pair(asset, factor_name)
                if factor_data is not None:
                    logging.info(f"Loaded factor data for iterative optimization: {len(factor_data)} rows")
            except Exception as e:
                logging.warning(f"Could not load factor data for {asset}/{factor_name}: {e}")
                factor_data = None
        
        # Run two-stage screening
        exploration_result = self.screening.run_complete_screening(
            asset=asset,
            factor_name=factor_name,
            param_grid=param_grid,
            date_range=date_range,
            historical_context=historical_context,
            factor_data=factor_data
        )
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("factor_exploration")
        
        return exploration_result
    
    def orchestrate_llm_analysis(self,
                                 asset: str,
                                 factor_name: str,
                                 backtest_result: Dict,
                                 use_batch: bool = False) -> Optional[Dict]:
        """
        Step 3: Orchestrate LLM analysis.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param backtest_result: Backtest result dictionary
        :param use_batch: Whether to use batch processing
        :return: LLM analysis result
        """
        if self.performance_monitor:
            self.performance_monitor.start_operation("llm_analysis")
        
        logging.info(f"Orchestrating LLM analysis for {asset}/{factor_name}")
        
        # Check if should use LLM
        should_use, reason = self.scheduler.should_use_llm(backtest_result)
        
        if not should_use:
            logging.info(f"Skipping LLM analysis: {reason}")
            return None
        
        # Prepare factor data
        factor_id = f"{asset}_{factor_name}"
        historical_context = self.context_manager.get_factor_summary(factor_id)
        
        factor_data = {
            'factor_name': factor_id,
            'backtest_results': backtest_result,
            'factor_params': backtest_result.get('params', {}),
            'historical_context': historical_context
        }
        
        # Request LLM analysis
        if use_batch:
            # Batch processing (for initial analysis)
            llm_result = self.scheduler.initial_analysis_batch([factor_data])
            if llm_result:
                return llm_result[0]
        else:
            # Individual processing (for optimization)
            llm_result = self.scheduler.optimization_round(factor_data)
            return llm_result
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("llm_analysis")
        
        return None
    
    def guide_optimization(self,
                          asset: str,
                          factor_name: str,
                          initial_result: Dict,
                          llm_analysis: Dict) -> Dict:
        """
        Step 4: Guide parameter optimization based on LLM suggestions.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param initial_result: Initial backtest result
        :param llm_analysis: LLM analysis result
        :return: Optimization results
        """
        if self.performance_monitor:
            self.performance_monitor.start_operation("parameter_optimization")
        
        logging.info(f"Guiding optimization for {asset}/{factor_name}")
        
        optimization_results = {
            'asset': asset,
            'factor_name': factor_name,
            'initial_result': initial_result,
            'llm_analysis': llm_analysis,
            'optimized_results': [],
            'best_optimized': None,
            'improvement': 0.0
        }
        
        # Extract optimization suggestions from LLM
        suggestions = self.screening._extract_optimization_suggestions(llm_analysis)
        
        if not suggestions:
            logging.warning("No optimization suggestions from LLM")
            return optimization_results
        
        # Test each suggestion
        for suggestion in suggestions[:3]:  # Test top 3 suggestions
            try:
                # Apply suggestion to parameters
                new_params = self.screening._apply_suggestion(
                    initial_result.get('params', {}),
                    suggestion
                )
                
                # Add required parameters
                new_params['initial_capital'] = self.strategy.initial_capital
                new_params['lot_size'] = self.strategy.min_lot_size
                if 'date_range' in initial_result.get('params', {}):
                    new_params['date_range'] = initial_result['params']['date_range']
                
                # Run backtest with new parameters
                result = self.strategy.run_backtest(asset, factor_name, new_params)
                
                if result:
                    # Calculate Calmar
                    annual_return = result.get('annual_return', 0)
                    max_drawdown = result.get('max_drawdown', 0)
                    calmar_ratio = self.screening._calculate_calmar_ratio(annual_return, max_drawdown)
                    result['calmar_ratio'] = calmar_ratio
                    result['params'] = new_params
                    result['suggestion'] = suggestion
                    
                    optimization_results['optimized_results'].append(result)
            
            except Exception as e:
                logging.error(f"Error testing optimization suggestion: {e}")
                continue
        
        # Find best optimized result
        if optimization_results['optimized_results']:
            best_optimized = max(
                optimization_results['optimized_results'],
                key=lambda x: x.get('sharpe_ratio', 0)
            )
            optimization_results['best_optimized'] = best_optimized
            
            initial_sharpe = initial_result.get('sharpe_ratio', 0)
            optimized_sharpe = best_optimized.get('sharpe_ratio', 0)
            optimization_results['improvement'] = optimized_sharpe - initial_sharpe
            
            logging.info(
                f"Optimization completed: Improvement {optimization_results['improvement']:.4f} "
                f"(from {initial_sharpe:.4f} to {optimized_sharpe:.4f})"
            )
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("parameter_optimization")
        
        return optimization_results
    
    def log_results(self,
                   asset: str,
                   factor_name: str,
                   exploration_result: Dict,
                   llm_analysis: Dict = None,
                   optimization_result: Dict = None):
        """
        Step 5: Log results to context manager.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param exploration_result: Exploration results
        :param llm_analysis: LLM analysis result (optional)
        :param optimization_result: Optimization result (optional)
        """
        if self.performance_monitor:
            self.performance_monitor.start_operation("result_logging")
        
        factor_id = f"{asset}_{factor_name}"
        logging.info(f"Logging results for {factor_id}")
        
        # Save factor context
        self.context_manager.save_factor_context(
            factor_id=factor_id,
            factor_name=factor_name,
            asset=asset,
            status='completed'
        )
        
        # Determine best result
        best_result = exploration_result.get('best_result')
        if optimization_result and optimization_result.get('best_optimized'):
            opt_sharpe = optimization_result['best_optimized'].get('sharpe_ratio', 0)
            exp_sharpe = best_result.get('sharpe_ratio', 0) if best_result else 0
            
            if opt_sharpe > exp_sharpe:
                best_result = optimization_result['best_optimized']
        
        if best_result:
            # Add analysis history
            self.context_manager.add_analysis_history(
                factor_id=factor_id,
                stage='exploration',
                params=best_result.get('params', {}),
                results=best_result,
                llm_analysis=llm_analysis
            )
            
            # Add optimization history if available
            if optimization_result and optimization_result.get('best_optimized'):
                initial_sharpe = exploration_result.get('best_result', {}).get('sharpe_ratio', 0)
                optimized_sharpe = optimization_result['best_optimized'].get('sharpe_ratio', 0)
                
                if optimized_sharpe > initial_sharpe:
                    self.context_manager.add_optimization_history(
                        factor_id=factor_id,
                        optimization_type='llm_guided',
                        before_params=exploration_result.get('best_result', {}).get('params', {}),
                        after_params=optimization_result['best_optimized'].get('params', {}),
                        before_sharpe=initial_sharpe,
                        after_sharpe=optimized_sharpe
                    )
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("result_logging")
        
        logging.info(f"Results logged for {factor_id}")
    
    def _count_param_combinations(self, param_grid: Dict) -> int:
        """Count parameter combinations in grid."""
        from itertools import product
        if not param_grid:
            return 0
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        return len(combinations)
    
    def process_factor(self,
                      asset: str,
                      factor_name: str,
                      param_grid: Dict,
                      date_range: Tuple[str, str] = None,
                      use_llm_optimization: bool = True) -> Dict:
        """
        Process a single factor through complete workflow.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param param_grid: Parameter grid for grid search
        :param date_range: Date range tuple or None
        :param use_llm_optimization: Whether to use LLM optimization
        :return: Complete processing results
        """
        from utils.log_config import should_log
        
        if should_log('info'):
            logging.info(f"Processing factor: {asset}/{factor_name}")
        
        result = {
            'asset': asset,
            'factor_name': factor_name,
            'factor_id': f"{asset}_{factor_name}",
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Check factor
            check_result = self.check_factor(asset, factor_name)
            result['steps']['check'] = check_result
            
            if check_result['status'] == 'failed':
                logging.warning(f"Factor check failed, skipping: {check_result['errors']}")
                result['status'] = 'failed'
                result['reason'] = 'Factor check failed'
                return result
            
            # Step 1.5: Generate intelligent parameters if param_grid not provided
            factor_data = check_result.get('factor_data')
            if param_grid is None and self.param_generator and factor_data is not None:
                logging.info("Generating intelligent parameter space using AI...")
                param_grid = self.param_generator.analyze_and_generate_params(
                    asset=asset,
                    factor_name=factor_name,
                    factor_data=factor_data
                )
                result['steps']['param_generation'] = {
                    'status': 'completed',
                    'param_grid': param_grid,
                    'combinations': self._count_param_combinations(param_grid)
                }
            
            # Step 2: Explore factor
            exploration_result = self.explore_factor(
                asset, 
                factor_name, 
                param_grid=param_grid, 
                date_range=date_range,
                factor_data=factor_data
            )
            result['steps']['exploration'] = exploration_result
            
            if exploration_result.get('status') == 'failed':
                logging.warning("Factor exploration failed")
                result['status'] = 'failed'
                result['reason'] = 'Exploration failed'
                return result
            
            best_result = exploration_result.get('best_result')
            if not best_result:
                logging.warning("No valid results from exploration")
                result['status'] = 'failed'
                result['reason'] = 'No valid results'
                return result
            
            # Step 3: LLM analysis
            llm_analysis = None
            if use_llm_optimization:
                llm_analysis = self.orchestrate_llm_analysis(
                    asset,
                    factor_name,
                    best_result,
                    use_batch=False
                )
                result['steps']['llm_analysis'] = llm_analysis
            
            # Step 4: Optimization (if LLM analysis available)
            optimization_result = None
            if llm_analysis and use_llm_optimization:
                optimization_result = self.guide_optimization(
                    asset,
                    factor_name,
                    best_result,
                    llm_analysis
                )
                result['steps']['optimization'] = optimization_result
            
            # Step 5: Log results
            self.log_results(
                asset,
                factor_name,
                exploration_result,
                llm_analysis,
                optimization_result
            )
            
            # Determine final best result
            final_best = best_result
            if optimization_result and optimization_result.get('best_optimized'):
                opt_sharpe = optimization_result['best_optimized'].get('sharpe_ratio', 0)
                exp_sharpe = best_result.get('sharpe_ratio', 0)
                
                if opt_sharpe > exp_sharpe:
                    final_best = optimization_result['best_optimized']
            
            result['best_result'] = final_best
            result['status'] = 'completed'
            
            if should_log('info'):
                logging.info(f"Factor processing completed: {asset}/{factor_name}")
                logging.info(f"  Best Sharpe: {final_best.get('sharpe_ratio', 0):.4f}")
                logging.info(f"  Best Calmar: {final_best.get('calmar_ratio', 0):.4f}")
        
        except Exception as e:
            logging.error(f"Error processing factor {asset}/{factor_name}: {e}")
            import traceback
            traceback.print_exc()
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result


if __name__ == "__main__":
    # Test the orchestrator
    print("Testing LLM Orchestrator...")
    
    # This would require full initialization of all components
    # For now, just demonstrate the structure
    print("Orchestrator structure created successfully!")
    print("Full testing requires initialization of all components.")
