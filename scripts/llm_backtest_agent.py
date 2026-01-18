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
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
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
from utils.log_config import configure_logging, set_verbosity

os.makedirs('logs', exist_ok=True)
log_file = f'logs/llm_agent_{datetime.now().strftime("%Y%m%d")}.log'

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
        
        # Strategy
        self.strategy = EnhancedNonPriceStrategy(
            initial_capital=self.config.get('initial_capital', 10000),
            min_lot_size=self.config.get('min_lot_size', 0.001),
            use_api=self.config.get('use_api', True),
            param_method=self.config.get('param_method', 'pct_change')
        )
        
        # LLM Client
        llm_config = self.config.get('llm', {})
        self.llm_client = LLMClient(
            api_url=llm_config.get('api_url', 'http://localhost:11434/api/chat'),
            model=llm_config.get('model', 'qwen2.5:3b')
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
            logging.info("✅ LLM connection successful")
        else:
            logging.error("❌ LLM connection failed")
            logging.error("Please ensure Ollama is running on Jetson")
        
        return success
    
    def get_factor_list(self) -> List[Dict]:
        """
        Get list of factors to analyze.
        
        :return: List of factor dictionaries
        """
        assets = self.config.get('assets', ['BTC'])
        factors = self.config.get('factors', [])
        max_factors = self.config.get('max_factors', None)  # Limit for testing
        
        factor_list = []
        
        for asset in assets:
            if factors:
                # Use specified factors
                for factor in factors:
                    factor_list.append({
                        'asset': asset,
                        'factor_name': factor
                    })
                    # Limit if max_factors specified
                    if max_factors and len(factor_list) >= max_factors:
                        break
            else:
                # Get all available factors for asset
                try:
                    available_factors = self.data_loader.get_available_factors(asset)
                    for factor in available_factors:
                        factor_list.append({
                            'asset': asset,
                            'factor_name': factor
                        })
                        # Limit if max_factors specified
                        if max_factors and len(factor_list) >= max_factors:
                            break
                except Exception as e:
                    logging.error(f"Error getting factors for {asset}: {e}")
            
            # Break if limit reached
            if max_factors and len(factor_list) >= max_factors:
                break
        
        logging.info(f"Found {len(factor_list)} factors to analyze")
        return factor_list
    
    def process_all_factors(self, factor_list: List[Dict]) -> List[Dict]:
        """
        Process all factors through complete workflow.
        
        :param factor_list: List of factors to process
        :return: List of processing results
        """
        logging.info(f"Starting to process {len(factor_list)} factors")
        
        results = []
        successful = 0
        failed = 0
        skipped = 0
        
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
        
        for i, factor_info in enumerate(factor_list, 1):
            asset = factor_info['asset']
            factor_name = factor_info['factor_name']
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing factor {i}/{len(factor_list)}: {asset}/{factor_name}")
            logging.info(f"{'='*80}")
            
            try:
                # Process factor
                result = self.orchestrator.process_factor(
                    asset=asset,
                    factor_name=factor_name,
                    param_grid=param_grid,
                    date_range=date_range,
                    use_llm_optimization=self.config.get('use_llm_optimization', True)
                )
                
                results.append(result)
                
                # Update counters
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
                
                # Periodic progress save
                if i % 50 == 0:
                    self._save_progress(results, i, len(factor_list))
            
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
        
        # Final summary
        from utils.log_config import should_log
        
        if should_log('info'):
            logging.info(f"\n{'='*80}")
            logging.info("PROCESSING SUMMARY")
            logging.info(f"{'='*80}")
            logging.info(f"Total factors: {len(factor_list)}")
            logging.info(f"Successful: {successful}")
            logging.info(f"Failed: {failed}")
            logging.info(f"Skipped: {skipped}")
            logging.info(f"{'='*80}")
        
        return results
    
    def _save_progress(self, results: List[Dict], current: int, total: int):
        """
        Save progress periodically.
        
        :param results: Current results
        :param current: Current progress
        :param total: Total factors
        """
        progress_file = f"reports/progress_{self.date_str}.json"
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'current': current,
            'total': total,
            'progress_pct': (current / total) * 100,
            'results': results
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, default=str)
        
        logging.info(f"Progress saved: {current}/{total} ({progress_data['progress_pct']:.1f}%)")
    
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
                'best_sharpe': 0.0
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
    
    def run_daily_workflow(self):
        """
        Run complete daily workflow.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"STARTING DAILY WORKFLOW - {self.date_str}")
        logging.info(f"{'='*80}\n")
        
        try:
            # Test LLM connection
            if not self.test_llm_connection():
                logging.warning("LLM connection failed, but continuing...")
            
            # Get factor list
            factor_list = self.get_factor_list()
            
            if not factor_list:
                logging.error("No factors to process!")
                return
            
            # Process all factors
            all_results = self.process_all_factors(factor_list)
            
            # Generate daily summary
            summary = self.generate_daily_summary(all_results)
            
            # Save to context manager
            self.context_manager.save_daily_summary(
                date=self.date_str,
                total_factors=summary['total_factors'],
                promising_factors=summary['promising_factors'],
                best_factor_id=summary['best_factor_id'],
                best_sharpe=summary['best_sharpe'],
                summary_data=summary['summary_data']
            )
            
            # Generate performance report
            self.performance_monitor.print_report()
            perf_report_file = f"reports/performance_{self.date_str}.json"
            os.makedirs(os.path.dirname(perf_report_file), exist_ok=True)
            self.performance_monitor.save_report(perf_report_file)
            
            # Get performance statistics
            perf_stats = self.performance_monitor.get_all_stats()
            
            # Generate daily report
            daily_report = self.report_generator.generate_daily_report(
                date_str=self.date_str,
                all_results=all_results,
                daily_summary=summary,
                performance_stats=perf_stats
            )
            
            # Save all reports
            self.report_generator.save_all_reports(daily_report, self.date_str)
            
            # Generate cache statistics
            if self.data_cache:
                self.data_cache.print_stats()
            
            logging.info(f"\n{'='*80}")
            logging.info(f"DAILY WORKFLOW COMPLETED - {self.date_str}")
            logging.info(f"{'='*80}")
            logging.info(f"Best Sharpe: {summary['best_sharpe']:.4f}")
            logging.info(f"Promising Factors: {summary['promising_factors']}/{summary['total_factors']}")
            logging.info(f"Reports saved to: reports/daily_reports/")
            
            return {
                'status': 'success',
                'date': self.date_str,
                'summary': summary,
                'report': daily_report,
                'results': all_results
            }
        
        except Exception as e:
            logging.error(f"Error in daily workflow: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'date': self.date_str,
                'error': str(e)
            }


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
        'initial_capital': 10000,
        'min_lot_size': 0.001,
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
            'api_url': 'http://localhost:11434/api/chat',
            'model': 'qwen2.5:3b'
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
    
    # Load configuration
    config = load_config(args.config)
    
    # Create agent
    agent = LLMBacktestAgent(config)
    
    # Test LLM connection if requested
    if args.test_llm:
        success = agent.test_llm_connection()
        sys.exit(0 if success else 1)
    
    # Run daily workflow
    result = agent.run_daily_workflow()
    
    if result.get('status') == 'success':
        logging.info("[OK] Daily workflow completed successfully!")
        sys.exit(0)
    else:
        logging.error("[ERROR] Daily workflow failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
