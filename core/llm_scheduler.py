"""
llm_scheduler.py

Intelligent LLM scheduler that decides when to call LLM and how to batch requests.

Features:
- Initial analysis: Batch processing (analyze multiple factors at once)
- Optimization stage: Individual processing (each factor optimized independently)
- Decision criteria: Only check Sharpe and Calmar ratios
- Stop conditions: Threshold reached or no improvement
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.llm_client import LLMClient
from core.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class IntelligentLLMScheduler:
    """
    Intelligent LLM scheduler with hybrid strategy.
    
    - Initial analysis: Batch process multiple factors
    - Optimization: Individual processing for each factor
    - Decision based on Sharpe and Calmar ratios
    """
    
    def __init__(self,
                 llm_client: LLMClient,
                 min_sharpe_threshold: float = 1.0,
                 min_calmar_threshold: float = 0.5,
                 max_optimization_rounds: int = 3,
                 min_improvement: float = 0.01,
                 batch_size: int = 10,
                 performance_monitor: PerformanceMonitor = None):
        """
        Initialize LLM scheduler.
        
        :param llm_client: LLM client instance
        :param min_sharpe_threshold: Minimum Sharpe ratio to use LLM
        :param min_calmar_threshold: Minimum Calmar ratio to use LLM
        :param max_optimization_rounds: Maximum optimization rounds per factor
        :param min_improvement: Minimum improvement to continue optimization
        :param batch_size: Batch size for initial analysis
        :param performance_monitor: Performance monitor instance
        """
        self.llm_client = llm_client
        self.min_sharpe_threshold = min_sharpe_threshold
        self.min_calmar_threshold = min_calmar_threshold
        self.max_optimization_rounds = max_optimization_rounds
        self.min_improvement = min_improvement
        self.batch_size = batch_size
        self.performance_monitor = performance_monitor
        
        logging.info("Intelligent LLM Scheduler initialized")
        logging.info(f"  Min Sharpe threshold: {min_sharpe_threshold}")
        logging.info(f"  Min Calmar threshold: {min_calmar_threshold}")
        logging.info(f"  Max optimization rounds: {max_optimization_rounds}")
        logging.info(f"  Batch size: {batch_size}")
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio.
        
        :param annual_return: Annual return (as decimal)
        :param max_drawdown: Maximum drawdown (as decimal)
        :return: Calmar ratio
        """
        if max_drawdown == 0 or abs(max_drawdown) < 0.001:
            return 0.0
        return annual_return / abs(max_drawdown)
    
    def should_use_llm(self, backtest_result: Dict) -> Tuple[bool, str]:
        """
        Decide whether to use LLM based on Sharpe and Calmar ratios.
        
        Decision criteria:
        - Sharpe ratio >= min_sharpe_threshold
        - Calmar ratio >= min_calmar_threshold
        
        :param backtest_result: Backtest result dictionary
        :return: Tuple (should_use_llm, reason)
        """
        sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
        
        # Calculate Calmar if not present
        if 'calmar_ratio' not in backtest_result:
            annual_return = backtest_result.get('annual_return', 0)
            max_drawdown = backtest_result.get('max_drawdown', 0)
            calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)
        else:
            calmar_ratio = backtest_result.get('calmar_ratio', 0)
        
        # Check Sharpe threshold
        if sharpe_ratio < self.min_sharpe_threshold:
            return False, f"Sharpe ratio {sharpe_ratio:.4f} < threshold {self.min_sharpe_threshold}"
        
        # Check Calmar threshold
        if calmar_ratio < self.min_calmar_threshold:
            return False, f"Calmar ratio {calmar_ratio:.4f} < threshold {self.min_calmar_threshold}"
        
        return True, f"Passed thresholds (Sharpe: {sharpe_ratio:.4f}, Calmar: {calmar_ratio:.4f})"
    
    def initial_analysis_batch(self, factors_data: List[Dict]) -> List[Dict]:
        """
        Initial analysis: Batch process multiple factors.
        
        Suitable for: One-time analysis of multiple factors to assess potential.
        
        :param factors_data: List of factor data dictionaries
        :return: List of LLM analysis results
        """
        logging.info(f"Initial batch analysis: Processing {len(factors_data)} factors")
        
        if self.performance_monitor:
            self.performance_monitor.start_operation("initial_analysis_batch")
        
        # Filter factors that should use LLM
        factors_to_analyze = []
        skipped_factors = []
        
        for factor_data in factors_data:
            backtest_result = factor_data.get('backtest_results', {})
            should_use, reason = self.should_use_llm(backtest_result)
            
            if should_use:
                factors_to_analyze.append(factor_data)
            else:
                skipped_factors.append({
                    'factor_name': factor_data.get('factor_name', 'Unknown'),
                    'reason': reason
                })
        
        logging.info(f"  {len(factors_to_analyze)} factors will be analyzed")
        logging.info(f"  {len(skipped_factors)} factors skipped")
        
        if not factors_to_analyze:
            logging.warning("No factors passed LLM criteria")
            return []
        
        # Process in batches
        all_results = []
        for i in range(0, len(factors_to_analyze), self.batch_size):
            batch = factors_to_analyze[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(factors_to_analyze) + self.batch_size - 1) // self.batch_size
            
            logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} factors)")
            
            # Batch LLM analysis
            batch_results = self.llm_client.analyze_batch(batch)
            all_results.extend(batch_results)
        
        if self.performance_monitor:
            self.performance_monitor.end_operation("initial_analysis_batch")
        
        logging.info(f"Batch analysis completed: {len(all_results)} results")
        
        return all_results
    
    def optimization_round(self,
                          factor_data: Dict,
                          previous_result: Dict = None,
                          round_number: int = 1) -> Dict:
        """
        Optimization stage: Individual processing for each factor.
        
        Suitable for: Each factor needs personalized optimization.
        
        :param factor_data: Factor data dictionary
        :param previous_result: Previous optimization result (for context)
        :param round_number: Current optimization round number
        :return: LLM analysis result
        """
        factor_name = factor_data.get('factor_name', 'Unknown')
        logging.info(f"Optimization round {round_number} for {factor_name}")
        
        if self.performance_monitor:
            self.performance_monitor.start_operation(f"optimization_round_{round_number}")
        
        # Add previous result to context if available
        if previous_result:
            historical_context = f"""
Previous optimization round {round_number - 1}:
- Sharpe ratio: {previous_result.get('sharpe_ratio', 0):.4f}
- Calmar ratio: {previous_result.get('calmar_ratio', 0):.4f}
- Parameters: {previous_result.get('params', {})}
"""
            factor_data['historical_context'] = factor_data.get('historical_context', '') + historical_context
        
        # Individual LLM analysis
        llm_result = self.llm_client.analyze_factor(factor_data)
        
        if self.performance_monitor:
            self.performance_monitor.end_operation(f"optimization_round_{round_number}")
        
        return llm_result
    
    def should_continue_optimization(self,
                                    factor_name: str,
                                    round_number: int,
                                    current_result: Dict,
                                    previous_results: List[Dict] = None) -> Tuple[bool, str]:
        """
        Decide whether to continue optimization.
        
        Stop conditions:
        1. Already excellent (Sharpe > 2.0 and Calmar > 3.0)
        2. No significant improvement
        3. Maximum rounds reached
        
        :param factor_name: Factor name
        :param round_number: Current round number
        :param current_result: Current optimization result
        :param previous_results: List of previous results
        :return: Tuple (should_continue, reason)
        """
        current_sharpe = current_result.get('sharpe_ratio', 0)
        current_calmar = current_result.get('calmar_ratio', 0)
        
        # Check if already excellent
        if current_sharpe > 2.0 and current_calmar > 3.0:
            return False, f"Already excellent (Sharpe: {current_sharpe:.4f}, Calmar: {current_calmar:.4f})"
        
        # Check improvement if we have previous results
        if previous_results and len(previous_results) > 0:
            previous_sharpe = previous_results[-1].get('sharpe_ratio', 0)
            improvement = current_sharpe - previous_sharpe
            
            if improvement < self.min_improvement:
                return False, f"No significant improvement (improvement: {improvement:.4f} < {self.min_improvement})"
        
        # Check maximum rounds
        if round_number >= self.max_optimization_rounds:
            return False, f"Maximum rounds reached ({self.max_optimization_rounds})"
        
        return True, f"Continue optimization (round {round_number}/{self.max_optimization_rounds})"
    
    def run_optimization_loop(self,
                             factor_data: Dict,
                             initial_result: Dict) -> Dict:
        """
        Run optimization loop for a single factor.
        
        :param factor_data: Factor data dictionary
        :param initial_result: Initial backtest result
        :return: Optimization results dictionary
        """
        factor_name = factor_data.get('factor_name', 'Unknown')
        logging.info(f"Starting optimization loop for {factor_name}")
        
        optimization_results = {
            'factor_name': factor_name,
            'initial_result': initial_result,
            'rounds': [],
            'final_result': None,
            'total_rounds': 0,
            'improvement': 0.0
        }
        
        current_result = initial_result
        previous_results = []
        
        for round_num in range(1, self.max_optimization_rounds + 1):
            # Check if should continue
            should_continue, reason = self.should_continue_optimization(
                factor_name,
                round_num,
                current_result,
                previous_results
            )
            
            if not should_continue:
                logging.info(f"Stopping optimization: {reason}")
                break
            
            # Run optimization round
            llm_analysis = self.optimization_round(
                factor_data,
                current_result,
                round_num
            )
            
            if not llm_analysis:
                logging.warning(f"LLM analysis failed in round {round_num}")
                break
            
            # Store round result
            round_result = {
                'round_number': round_num,
                'llm_analysis': llm_analysis,
                'result_before': current_result.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            optimization_results['rounds'].append(round_result)
            previous_results.append(current_result)
            
            # Note: In real implementation, you would apply LLM suggestions
            # and run new backtest here to get updated current_result
            # For now, we just track the optimization rounds
            
            logging.info(f"Round {round_num} completed for {factor_name}")
        
        optimization_results['total_rounds'] = len(optimization_results['rounds'])
        
        if optimization_results['rounds']:
            optimization_results['final_result'] = current_result
            initial_sharpe = initial_result.get('sharpe_ratio', 0)
            final_sharpe = current_result.get('sharpe_ratio', 0)
            optimization_results['improvement'] = final_sharpe - initial_sharpe
        
        logging.info(f"Optimization loop completed for {factor_name}: {optimization_results['total_rounds']} rounds")
        
        return optimization_results


if __name__ == "__main__":
    # Test the LLM scheduler
    from core.llm_client import LLMClient
    
    print("Testing Intelligent LLM Scheduler...")
    
    # Initialize LLM client
    llm_client = LLMClient(api_url="http://localhost:11434/api/chat")
    
    # Create scheduler
    scheduler = IntelligentLLMScheduler(
        llm_client=llm_client,
        min_sharpe_threshold=1.0,
        min_calmar_threshold=0.5,
        max_optimization_rounds=3,
        batch_size=10
    )
    
    # Test should_use_llm
    print("\n1. Testing should_use_llm...")
    test_result_good = {
        'sharpe_ratio': 1.5,
        'annual_return': 0.25,
        'max_drawdown': -0.10,
        'calmar_ratio': 2.5
    }
    
    test_result_bad = {
        'sharpe_ratio': 0.3,
        'annual_return': 0.10,
        'max_drawdown': -0.20,
        'calmar_ratio': 0.5
    }
    
    should_use_1, reason_1 = scheduler.should_use_llm(test_result_good)
    should_use_2, reason_2 = scheduler.should_use_llm(test_result_bad)
    
    print(f"  Good result: {should_use_1} - {reason_1}")
    print(f"  Bad result: {should_use_2} - {reason_2}")
    
    # Test should_continue_optimization
    print("\n2. Testing should_continue_optimization...")
    excellent_result = {
        'sharpe_ratio': 2.5,
        'calmar_ratio': 3.5
    }
    
    should_continue, reason = scheduler.should_continue_optimization(
        "test_factor",
        1,
        excellent_result
    )
    print(f"  Excellent result: {should_continue} - {reason}")
    
    print("\n✅ LLM Scheduler tests completed!")
