"""
enhanced_engine.py

Enhanced backtesting engine with realistic trading simulation.

Features:
- Dynamic lot size calculation based on available capital
- Position tracking with pos_opened and pnl_list
- Proper position closing and opening for direction changes
- More realistic trading simulation
"""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Union, Callable
import math
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import itertools
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent Qt errors
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from datetime import datetime
from config.trading_config import DEFAULT_INITIAL_CAPITAL

# Set deterministic behavior for reproducible results
np.random.seed(42)
pd.set_option('display.float_format', lambda x: '%.8f' % x)


@dataclass
class Position:
    """Position tracking structure"""
    timestamp: datetime
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    entry_capital: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_open: bool = True


@dataclass
class Trade:
    """Trade execution structure"""
    timestamp: datetime
    type: str  # 'open_long', 'close_long', 'open_short', 'close_short'
    price: float
    quantity: float
    capital_used: float
    pnl: float = 0.0


class EnhancedBacktestEngine(ABC):
    """
    Enhanced backtesting engine with realistic trading simulation.
    
    Features:
    - Dynamic position sizing based on available capital
    - Position tracking with pos_opened and pnl_list
    - Proper position closing and opening for direction changes
    - More realistic trading simulation
    
    :param initial_capital: Starting capital for backtests
    :param max_workers: Maximum parallel processes for optimization
    :param position_size_pct: Percentage of capital to use per position (default: 0.1 = 10%)
    :param min_lot_size: Minimum lot size for trading
    :precondition: Subclasses must implement generate_signal()
    :postcondition: Realistic trading simulation with proper position management
    """
    
    def __init__(self, 
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL, 
                 max_workers: int = 4,
                 position_size_pct: float = 0.1,
                 min_lot_size: float = 0.001):
        self.initial_capital = initial_capital
        self.max_workers = max_workers
        self.position_size_pct = position_size_pct
        self.min_lot_size = min_lot_size
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure standardized logging format"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @abstractmethod
    def generate_signal(self, row: pd.Series, params: Dict) -> int:
        """
        Generate trading signal from data row
        
        :param row: Single row of market data
        :param params: Strategy parameters
        :return: 1 (long), -1 (short), 0 (hold)
        """
        pass

    def calculate_lot_size(self, available_capital: float, price: float, lot_size: float = None) -> float:
        """
        Calculate lot size based on available capital and price.
        
        Logic: Use all available capital to buy as many lots as possible,
        but ensure minimum lot size is met.
        
        :param available_capital: Available capital for trading
        :param price: Current price
        :param lot_size: Minimum lot size (if None, uses self.min_lot_size)
        :return: Calculated lot size
        """
        if lot_size is None:
            lot_size = self.min_lot_size
        
        # Calculate how many lots we can buy with available capital
        max_lots = available_capital / price
        
        # Round down to ensure we don't exceed available capital
        # and ensure minimum lot size
        calculated_lot_size = max(lot_size, max_lots)
        
        # Ensure we don't exceed available capital
        if calculated_lot_size * price > available_capital:
            calculated_lot_size = available_capital / price
        
        return calculated_lot_size

    def execute_backtest(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Execute single backtest with exact legacy logic.
        
        :param data: Historical data with 'timestamp', 'price', 'last_price', 'rolling_return'
        :param params: Strategy parameters dictionary
        :return: Dictionary with results and metrics
        """
        # EXACT same variables as legacy logic - RESET FOR EACH CALL
        pos_opened = False  # Boolean flag, not a list
        pnl_list = []  # Empty list like legacy logic - RESET FOR EACH CALL
        equity_value_list = []
        date_list = []
        equity_value_pnl_pct_list = []
        
        original_capital = params.get('initial_capital', 10000)
        lot_size = params.get('lot_size', 0.001)
        num_of_lot = 0
        open_price = 0
        temp_capital = original_capital
        last_direction = 0
        
        # Get parameters exactly like legacy logic
        long_param = params.get('long_param', 0.02)
        short_param = params.get('short_param', -0.02)
        rolling = params.get('rolling', 1)
        

        

        
        data = data.copy()

        # EXACT direction/signal processing from legacy logic
        data['direction'] = 0  # Initialize with 0
        
        # EXACT same logic as legacy logic:
        data.loc[(data['rolling_return'] < long_param), 'direction'] = 1
        data.loc[(data['rolling_return'] > short_param), 'direction'] = -1
        
        # EXACT signal logic from legacy logic:
        data['signal'] = False  # Initialize with False
        data.loc[data['direction'] == 1, 'signal'] = True
        data.loc[data['direction'] == -1, 'signal'] = True
        
        # EXACT fillna from legacy logic:
        data['direction'].fillna(0, inplace=True)
        data['signal'].fillna(False, inplace=True)

        # EXACT same loop as legacy logic
        for i in range(len(data)):
            current = data.iloc[i]
            
            # Get current values (EXACT same naming as legacy logic)
            now_date = current['timestamp']
            now_close_crypto = current['price']
            now_close_crypto_last_day = current['last_price']
            
            # Get yesterday's position direction (EXACT same as legacy logic)
            if i != 0:
                position_last_day = data.iloc[i-1]['direction']
            else:
                position_last_day = current['direction']
            
            signal = current['signal']
            direction = current['direction']
            
            # EXACT equity curve calculation from legacy logic
            equity_value = temp_capital + (now_close_crypto - open_price) * num_of_lot * lot_size
            equity_value_list.append(equity_value)
            date_list.append(now_date)
            
            # EXACT P&L calculation from legacy logic
            equity_value_pnl = (now_close_crypto - open_price) * num_of_lot * lot_size
            
            # EXACT percentage P&L using yesterday's close price from legacy logic
            if now_close_crypto_last_day and not pd.isna(now_close_crypto_last_day) and now_close_crypto_last_day != 0:
                equity_value_pnl_pct = ((now_close_crypto - now_close_crypto_last_day) / now_close_crypto_last_day) * position_last_day
            else:
                equity_value_pnl_pct = 0.0
            
            equity_value_pnl_pct_list.append(equity_value_pnl_pct)
            
            # EXACT position open logic from legacy logic
            if (pos_opened == False) and (signal == True):
                pos_opened = True
                open_price = now_close_crypto
                num_of_lot = temp_capital // (open_price * lot_size)  # Integer division
                last_direction = direction

            
            # EXACT position change logic from legacy logic
            elif (pos_opened == True) and (signal == True) and (last_direction != direction):
                pos_opened = False
                close_price = now_close_crypto
                pnl = (close_price - open_price) * num_of_lot * lot_size * last_direction
                pnl_list.append(pnl)  # EXACT same as legacy logic

                # Debug: Log position change (only first 3 and last 3 trades)
                if len(pnl_list) <= 3 or len(pnl_list) >= len(data) - 3:
                    logging.info(f"Position Change - Date: {now_date}, Open: {open_price:.2f}, Close: {close_price:.2f}, "
                               f"Direction: {last_direction}, Lots: {num_of_lot}, PnL: {pnl:.2f}, Capital: {temp_capital:.2f}")
                
                temp_capital = temp_capital + pnl
                open_price = 0
                num_of_lot = 0
                
                pos_opened = True
                open_price = now_close_crypto
                num_of_lot = temp_capital // (open_price * lot_size)
                last_direction = direction
            
            # EXACT position close logic from legacy logic
            elif (pos_opened == True) and ((signal == False) or (i == len(data) - 1)):
                pos_opened = False
                close_price = now_close_crypto
                pnl = (close_price - open_price) * num_of_lot * lot_size * last_direction
                pnl_list.append(pnl)  # EXACT same as legacy logic
                
                # Debug: Log position close (only first 3 and last 3 trades)
                if len(pnl_list) <= 3 or len(pnl_list) >= len(data) - 3:
                    logging.info(f"Position Close - Date: {now_date}, Open: {open_price:.2f}, Close: {close_price:.2f}, "
                               f"Direction: {last_direction}, Lots: {num_of_lot}, PnL: {pnl:.2f}, Capital: {temp_capital:.2f}")
                
                temp_capital = temp_capital + pnl
                open_price = 0
                num_of_lot = 0
        
        # EXACT same calculation as legacy logic
        if len(pnl_list) > 0:
            total_profit = temp_capital - original_capital  # Total profit including unrealized P&L
            num_of_trade = len(pnl_list)  # EXACT same as legacy logic
            
            # Debug: Log key calculation values
            logging.info(f"Backtest Debug - Original Capital: {original_capital:.2f}")
            logging.info(f"Backtest Debug - Final Capital: {temp_capital:.2f}")
            logging.info(f"Backtest Debug - Total Profit: {total_profit:.2f}")
            logging.info(f"Backtest Debug - Number of Trades: {num_of_trade}")
            logging.info(f"Backtest Debug - PnL List Sum: {sum(pnl_list):.2f}")
            
            # Remove first 'rolling' elements like legacy logic
            if len(equity_value_pnl_pct_list) > rolling:
                del equity_value_pnl_pct_list[:rolling]
            
            if len(equity_value_pnl_pct_list) > 0:
                avg_return = sum(equity_value_pnl_pct_list) / len(equity_value_pnl_pct_list)
                annual_return = avg_return * 365
                return_sd = np.std(equity_value_pnl_pct_list, ddof=0)
                sharpe = avg_return / return_sd * math.sqrt(365) if return_sd > 0 else 0
            else:
                avg_return = 0
                annual_return = 0
                sharpe = 0
        else:
            total_profit = 0
            num_of_trade = 0
            avg_return = 0
            annual_return = 0
            sharpe = 0
        
        # Calculate total return based on final capital (like legacy logic)
        total_return = (temp_capital - original_capital) / original_capital if original_capital > 0 else 0
        
        # Calculate max drawdown using dollar amounts (like legacy logic)
        if equity_value_list:
            peak = equity_value_list[0]
            max_dd_dollar = 0
            max_dd_pct = 0
            for value in equity_value_list:
                if value > peak:
                    peak = value
                dd_dollar = peak - value
                dd_pct = (peak - value) / peak if peak > 0 else 0
                max_dd_dollar = max(max_dd_dollar, dd_dollar)
                max_dd_pct = max(max_dd_pct, dd_pct)
        else:
            max_dd_dollar = 0
            max_dd_pct = 0
        

        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd_pct,  # Percentage-based max drawdown
            'max_drawdown_dollar': max_dd_dollar,  # Dollar-based max drawdown
            'win_rate': 0.0,  # Not calculated in original
            'num_trades': num_of_trade,  # EXACT same as legacy logic
            'num_positions': 0,  # Not tracked in original
            'avg_position_duration': 0.0,
            'profit_factor': 0.0,
            'max_consecutive_losses': 0,
            'equity_curve': equity_value_list,
            'trades': [],  # Not tracked in original
            'pos_opened': [],  # Not tracked in original
            'params': params,  # Store parameters used
            'final_capital': temp_capital,
            'total_profit': total_profit,  # Only closed trades profit
            'pnl_list': pnl_list,  # Store PnL list for debugging
            'data': data  # Store data for Buy & Hold calculation
        }

    def _calculate_performance(self, equity: List[float], trades: List[Trade], 
                             pos_opened: List[Position], pnl_list: List[float], 
                             pnl_pct_list: List[float] = None) -> Dict:
        """
        Calculate comprehensive performance metrics with yesterday's close price logic
        
        :param equity: List of equity values over time
        :param trades: List of trade objects
        :param pos_opened: List of opened positions
        :param pnl_list: List of P&L values over time
        :param pnl_pct_list: List of percentage P&L values over time (yesterday's close based)
        :return: Dictionary of performance metrics
        """
        if not equity:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'num_positions': 0,
                'avg_position_duration': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_losses': 0,
                'pnl_pct_metrics': {}
            }
        
        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Annualized return (assuming daily data)
        days = len(equity)
        annual_return = ((equity[-1] / equity[0]) ** (365 / days)) - 1 if days > 0 else 0
        
        # Sharpe ratio
        returns = pd.Series(equity).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * math.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Win rate and profit factor
        win_rate, profit_factor, max_consecutive_losses = self._calculate_trade_metrics(trades)
        
        # Position metrics
        num_positions = len(pos_opened)
        avg_position_duration = self._calculate_avg_position_duration(pos_opened)
        
        # Calculate percentage P&L metrics (yesterday's close based)
        pnl_pct_metrics = {}
        if pnl_pct_list and len(pnl_pct_list) > 0:
            pnl_pct_series = pd.Series(pnl_pct_list)
            pnl_pct_metrics = {
                'total_pnl_pct': pnl_pct_series.sum(),
                'avg_pnl_pct': pnl_pct_series.mean(),
                'max_pnl_pct': pnl_pct_series.max(),
                'min_pnl_pct': pnl_pct_series.min(),
                'pnl_pct_std': pnl_pct_series.std(),
                'positive_pnl_pct_days': (pnl_pct_series > 0).sum(),
                'negative_pnl_pct_days': (pnl_pct_series < 0).sum(),
                'pnl_pct_win_rate': (pnl_pct_series > 0).mean() if len(pnl_pct_series) > 0 else 0
            }
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'num_positions': num_positions,
            'avg_position_duration': avg_position_duration,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'pnl_pct_metrics': pnl_pct_metrics
        }

    def _calculate_trade_metrics(self, trades: List[Trade]) -> Tuple[float, float, int]:
        """
        Calculate trade-based metrics
        
        :param trades: List of trade objects
        :return: Tuple of (win_rate, profit_factor, max_consecutive_losses)
        """
        if len(trades) == 0:
            return 0.0, 0.0, 0
        
        # Separate close trades (which have P&L)
        close_trades = [t for t in trades if t.type.startswith('close')]
        
        if len(close_trades) == 0:
            return 0.0, 0.0, 0
        
        # Calculate win rate
        winning_trades = [t for t in close_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(close_trades)
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        losing_trades = [t for t in close_trades if t.pnl < 0]
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max consecutive losses
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        for trade in close_trades:
            if trade.pnl < 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0
        
        return win_rate, profit_factor, max_consecutive_losses

    def _calculate_avg_position_duration(self, pos_opened: List[Position]) -> float:
        """
        Calculate average position duration
        
        :param pos_opened: List of opened positions
        :return: Average duration in days
        """
        if not pos_opened:
            return 0.0
        
        total_duration = 0
        count = 0
        
        for pos in pos_opened:
            if not pos.is_open:  # Only count closed positions
                # Calculate duration (simplified - assumes daily data)
                duration = 1  # Placeholder - would need actual timestamps
                total_duration += duration
                count += 1
        
        return total_duration / count if count > 0 else 0.0

    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        param_grid: Dict[str, List[Any]],
        objective: str = 'sharpe_ratio',
        output_dir: str = 'reports'
    ) -> Dict:
        """
        Perform parallel parameter optimization
        
        :param data: Historical data for backtesting
        :param param_grid: Parameter search space
        :param objective: Optimization metric (sharpe_ratio, total_return, etc.)
        :param output_dir: Directory for optimization reports
        :return: Best result dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Parallel backtest execution
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for params in param_combinations:
                futures.append(executor.submit(
                    self._evaluate_params, 
                    data.copy(), 
                    params
                ))
            
            results = []
            for future in tqdm(futures, desc="Optimizing parameters"):
                results.append(future.result())
        
        # Identify best result
        best_result = max(results, key=lambda x: x[objective])
        
        # Generate reports
        self._generate_optimization_report(
            results, 
            best_result, 
            output_dir,
            objective
        )
        
        return best_result

    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        keys = param_grid.keys()
        values = param_grid.values()
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _evaluate_params(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Wrapper for parallel parameter evaluation"""
        return self.execute_backtest(data, params)

    def _generate_optimization_report(
        self,
        results: List[Dict],
        best_result: Dict,
        output_dir: str,
        objective: str
    ):
        """
        Generate comprehensive optimization report with standardized output format.
        
        :param results: List of all optimization results
        :param best_result: Best performing result
        :param output_dir: Output directory for reports
        :param objective: Optimization objective used
        """
        # Create summary report
        summary_file = os.path.join(output_dir, 'optimization_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Optimization Summary\n")
            f.write(f"===================\n")
            f.write(f"Objective: {objective}\n")
            f.write(f"Total combinations tested: {len(results)}\n")
            f.write(f"Best {objective}: {best_result[objective]:.4f}\n")
            f.write(f"Best parameters: {best_result['params']}\n\n")
            
            # Top 10 results
            f.write(f"Top 10 Results:\n")
            sorted_results = sorted(results, key=lambda x: x[objective], reverse=True)
            for i, result in enumerate(sorted_results[:10]):
                f.write(f"{i+1}. {objective}: {result[objective]:.4f}, Params: {result['params']}\n")
        
        # Generate standardized results table
        self._generate_results_table(results, output_dir)
        
        # Generate buy & hold summary
        self._generate_buy_hold_summary(results, output_dir)
        
        # Create 2x2 plots
        self._create_standardized_plots(results, best_result, output_dir, objective)
        
        logging.info(f"Optimization report saved to {output_dir}")

    def _generate_results_table(self, results: List[Dict], output_dir: str):
        """
        Generate standardized results table with required columns.
        
        :param results: List of optimization results
        :param output_dir: Output directory
        """
        # Prepare data for table
        table_data = []
        
        for result in results:
            params = result.get('params', {})
            
            # Extract parameters - FIX: Use correct parameter names
            rolling = params.get('rolling', 'N/A')
            long_param = params.get('long_param', 'N/A')  # FIX: was 'long_threshold'
            short_param = params.get('short_param', 'N/A')  # FIX: was 'short_threshold'
            param_method = params.get('param_method', 'N/A')
            
            # Extract metrics
            total_profit = result.get('total_profit', 0)  # Use actual dollar profit amount
            num_of_trade = result.get('num_trades', 0)
            
            # Calculate MDD
            equity_curve = result.get('equity_curve', [])
            mdd_dollar, mdd_pct = self._calculate_mdd(equity_curve)
            
            # Calculate other metrics
            annual_return = result.get('annual_return', 0) * 100  # Convert to percentage
            sharpe = result.get('sharpe_ratio', 0)
            
            # Calculate Calmar ratio
            calmar_ratio = annual_return / abs(mdd_pct) if mdd_pct != 0 else 0
            
            # Final equity
            final_equity = equity_curve[-1] if equity_curve else 0
            
            table_data.append({
                'rolling': rolling,
                'long_param': long_param,
                'short_param': short_param,
                'param_method': param_method,
                'total_profit': total_profit,
                'num_of_trade': num_of_trade,
                'mdd_dollar': mdd_dollar,
                'mdd_pct': mdd_pct,
                'annual_return': annual_return,
                'calmar_ratio': calmar_ratio,
                'sharpe': sharpe,
                'final_equity': final_equity
            })
        
        # Sort by Sharpe ratio (descending)
        table_data.sort(key=lambda x: x['sharpe'], reverse=True)
        
        # Print results table
        print("\n" + "="*120)
        print("PARAMETER SEARCH RESULTS TABLE")
        print("="*120)
        
        # Print header
        header = f"{'rolling':<8} {'long_param':<12} {'short_param':<13} {'param_method':<12} "
        header += f"{'total_profit':<12} {'num_of_trade':<12} {'mdd_dollar':<12} {'mdd_pct':<10} "
        header += f"{'annual_return':<12} {'calmar_ratio':<12} {'sharpe':<8} {'final_equity':<12}"
        print(header)
        print("-" * 120)
        
        # Print data rows
        for row in table_data:
            line = f"{row['rolling']:<8} {row['long_param']:<12.4f} {row['short_param']:<13.4f} {row['param_method']:<12} "
            line += f"{row['total_profit']:<12.2f} {row['num_of_trade']:<12} {row['mdd_dollar']:<12.2f} {row['mdd_pct']:<10.2f} "
            line += f"{row['annual_return']:<12.2f} {row['calmar_ratio']:<12.2f} {row['sharpe']:<8.2f} {row['final_equity']:<12.2f}"
            print(line)
        
        print("="*120)
        
        # Save to CSV
        df_results = pd.DataFrame(table_data)
        csv_file = os.path.join(output_dir, 'optimization_results.csv')
        df_results.to_csv(csv_file, index=False)
        logging.info(f"Results table saved to {csv_file}")

    def _generate_buy_hold_summary(self, results: List[Dict], output_dir: str):
        """
        Generate buy & hold summary using EXACT logic from legacy algorithm.
        
        :param results: List of optimization results
        :param output_dir: Output directory
        """
        if not results:
            return
        
        # Get price data from the first result's data
        first_result = results[0]
        data = first_result.get('data', None)
        
        if data is None or len(data) == 0:
            logging.warning("No data available for buy & hold calculation")
            return
        
        # EXACT same logic as legacy algorithm for Buy & Hold
        # Get price data (assuming 'price' column exists)
        if 'price' not in data.columns:
            logging.warning("Price column not found in data")
            return
        
        price_data = data['price'].values.tolist()
        
        # Calculate percentage changes (EXACT same as legacy algorithm)
        bnh_pct_change_list = []
        for i in range(1, len(price_data)):
            if price_data[i-1] != 0:
                pct_change = (price_data[i] - price_data[i-1]) / price_data[i-1]
                bnh_pct_change_list.append(pct_change)
        
        # Remove NaN values (EXACT same as legacy algorithm)
        bnh_pct_change_list = [item for item in bnh_pct_change_list if not (math.isnan(item)) == True]
        
        # Calculate price changes (EXACT same as legacy algorithm)
        bnh_change_list = []
        for i in range(1, len(price_data)):
            change = price_data[i] - price_data[i-1]
            bnh_change_list.append(change)
        
        # Calculate cumulative changes (EXACT same as legacy algorithm)
        bnh_change_accu = np.cumsum(bnh_change_list)
        
        # Calculate metrics (EXACT same as legacy algorithm)
        if len(bnh_pct_change_list) > 0:
            bnh_avg_return = sum(bnh_pct_change_list) / len(bnh_pct_change_list)
            bnh_annual_return = bnh_avg_return * 365
            bnh_return_sd = np.std(bnh_pct_change_list, ddof=0)
            sharpe_bnh = bnh_avg_return / bnh_return_sd * math.sqrt(365) if bnh_return_sd > 0 else 0
        else:
            bnh_avg_return = 0
            bnh_annual_return = 0
            sharpe_bnh = 0
        
        # Calculate MDD using price data (EXACT same as legacy algorithm)
        equity_value_bnh_list = []
        dd_dollar_bnh_list = []
        dd_pct_bnh_list = []
        
        for i in range(len(price_data)):
            equity_value_bnh = price_data[i]
            equity_value_bnh_list.append(equity_value_bnh)
            
            # Calculate drawdown
            temp_max_equity_bnh = max(equity_value_bnh_list)
            dd_dollar_bnh = temp_max_equity_bnh - equity_value_bnh
            dd_dollar_bnh_list.append(dd_dollar_bnh)
            
            dd_pct_bnh = (temp_max_equity_bnh - equity_value_bnh) / temp_max_equity_bnh
            dd_pct_bnh_list.append(dd_pct_bnh)
        
        # Get maximum drawdowns
        mdd_dollar_bnh = max(dd_dollar_bnh_list) if dd_dollar_bnh_list else 0
        mdd_pct_bnh = max(dd_pct_bnh_list) if dd_pct_bnh_list else 0
        
        # Calculate Calmar ratio
        Calmar_Ratio_bnh = bnh_annual_return / mdd_pct_bnh if mdd_pct_bnh != 0 else 0
        
        # Print buy & hold summary (EXACT same format as legacy algorithm)
        print("\n" + "="*60)
        print("BUY & HOLD SUMMARY")
        print("="*60)
        print(f"MDD Dollar: ${mdd_dollar_bnh:.2f}")
        print(f"MDD %: {mdd_pct_bnh:.2%}")
        print(f"Annual Return: {bnh_annual_return:.2%}")
        print(f"Calmar Ratio: {Calmar_Ratio_bnh:.2f}")
        print(f"Sharpe: {sharpe_bnh:.2f}")
        print("="*60)
        
        # Save to file
        bh_file = os.path.join(output_dir, 'buy_hold_summary.txt')
        with open(bh_file, 'w') as f:
            f.write("BUY & HOLD SUMMARY\n")
            f.write("==================\n")
            f.write(f"MDD Dollar: ${mdd_dollar_bnh:.2f}\n")
            f.write(f"MDD %: {mdd_pct_bnh:.2%}\n")
            f.write(f"Annual Return: {bnh_annual_return:.2%}\n")
            f.write(f"Calmar Ratio: {Calmar_Ratio_bnh:.2f}\n")
            f.write(f"Sharpe: {sharpe_bnh:.2f}\n")

    def _calculate_mdd(self, equity_curve: List[float]) -> Tuple[float, float]:
        """
        Calculate Maximum Drawdown in dollars and percentage.
        
        :param equity_curve: List of equity values
        :return: Tuple of (mdd_dollar, mdd_pct)
        """
        if not equity_curve:
            return 0.0, 0.0
        
        peak = equity_curve[0]
        max_dd_dollar = 0
        max_dd_pct = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd_dollar = peak - value
            dd_pct = (dd_dollar / peak) * 100 if peak > 0 else 0
            
            if dd_dollar > max_dd_dollar:
                max_dd_dollar = dd_dollar
                max_dd_pct = dd_pct
        
        return max_dd_dollar, max_dd_pct

    def _create_standardized_plots(self, results: List[Dict], best_result: Dict, 
                                 output_dir: str, objective: str):
        """
        Create 2x2 plots as specified in requirements.
        
        :param results: List of optimization results
        :param best_result: Best performing result
        :param output_dir: Output directory
        :param objective: Optimization objective
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Optimization Analysis - {objective.replace("_", " ").title()}', fontsize=16)
            
            # 1. Strategy vs Buy & Hold (equity curves)
            ax1 = axes[0, 0]
            if best_result.get('equity_curve'):
                strategy_curve = best_result['equity_curve']
                ax1.plot(strategy_curve, label='Strategy', linewidth=2)
                
                # Buy & hold curve (simplified - using first result's data)
                if results:
                    bh_curve = results[0].get('equity_curve', [])
                    if bh_curve:
                        ax1.plot(bh_curve, label='Buy & Hold', linewidth=2, alpha=0.7)
                
                ax1.set_title(f'Strategy vs Buy & Hold\nSharpe: {best_result.get("sharpe_ratio", 0):.2f}, '
                            f'Annual: {best_result.get("annual_return", 0)*100:.1f}%')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Equity')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Trade distribution (Long vs Short counts)
            ax2 = axes[0, 1]
            long_counts = []
            short_counts = []
            
            for result in results:
                pos_opened = result.get('pos_opened', [])
                long_count = len([p for p in pos_opened if p.side == 'long'])
                short_count = len([p for p in pos_opened if p.side == 'short'])
                long_counts.append(long_count)
                short_counts.append(short_count)
            
            if long_counts and short_counts:
                ax2.hist(long_counts, alpha=0.7, label='Long Trades', bins=20)
                ax2.hist(short_counts, alpha=0.7, label='Short Trades', bins=20)
                ax2.set_title('Trade Distribution')
                ax2.set_xlabel('Number of Trades')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. Performance comparison (above median)
            ax3 = axes[1, 0]
            sharpe_values = [r.get('sharpe_ratio', 0) for r in results]
            median_sharpe = np.median(sharpe_values) if sharpe_values else 0
            
            above_median = [r for r in results if r.get('sharpe_ratio', 0) > median_sharpe]
            below_median = [r for r in results if r.get('sharpe_ratio', 0) <= median_sharpe]
            
            if above_median and below_median:
                above_returns = [r.get('total_return', 0) for r in above_median]
                below_returns = [r.get('total_return', 0) for r in below_median]
                
                ax3.boxplot([above_returns, below_returns], 
                           labels=['Above Median Sharpe', 'Below Median Sharpe'])
                ax3.set_title('Performance Comparison')
                ax3.set_ylabel('Total Return')
                ax3.grid(True, alpha=0.3)
            
            # 4. Sharpe heatmap across parameter grid
            ax4 = axes[1, 1]
            if results and 'params' in results[0]:
                param_names = list(results[0]['params'].keys())
                if len(param_names) >= 2:
                    # Create parameter grid
                    param1_values = sorted(list(set(r['params'].get(param_names[0], 0) for r in results)))
                    param2_values = sorted(list(set(r['params'].get(param_names[1], 0) for r in results)))
                    
                    # Create heatmap data
                    heatmap_data = np.zeros((len(param2_values), len(param1_values)))
                    
                    for i, p2 in enumerate(param2_values):
                        for j, p1 in enumerate(param1_values):
                            # Find matching result
                            matching_results = [r for r in results 
                                              if r['params'].get(param_names[0]) == p1 
                                              and r['params'].get(param_names[1]) == p2]
                            if matching_results:
                                heatmap_data[i, j] = matching_results[0].get('sharpe_ratio', 0)
                    
                    im = ax4.imshow(heatmap_data, cmap='viridis', aspect='auto')
                    ax4.set_xticks(range(len(param1_values)))
                    ax4.set_yticks(range(len(param2_values)))
                    ax4.set_xticklabels([f'{p:.3f}' for p in param1_values], rotation=45)
                    ax4.set_yticklabels([f'{p:.3f}' for p in param2_values])
                    ax4.set_xlabel(param_names[0])
                    ax4.set_ylabel(param_names[1])
                    ax4.set_title('Sharpe Ratio Heatmap')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax4, label='Sharpe Ratio')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'optimization_analysis_2x2.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"2x2 plots saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Failed to create standardized plots: {e}")

    def _create_optimization_plots(self, results: List[Dict], best_result: Dict, 
                                 output_dir: str, objective: str):
        """Create optimization visualization plots (legacy method - kept for compatibility)"""
        try:
            # Extract objective values
            objective_values = [r[objective] for r in results]
            
            # Create histogram of results
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(objective_values, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(best_result[objective], color='red', linestyle='--', 
                       label=f'Best: {best_result[objective]:.4f}')
            plt.xlabel(objective.replace('_', ' ').title())
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {objective.replace("_", " ").title()}')
            plt.legend()
            
            # Create scatter plot of key parameters vs objective
            if len(results) > 0 and 'params' in results[0]:
                param_names = list(results[0]['params'].keys())
                if len(param_names) >= 2:
                    plt.subplot(2, 2, 2)
                    param1_values = [r['params'].get(param_names[0], 0) for r in results]
                    param2_values = [r['params'].get(param_names[1], 0) for r in results]
                    
                    scatter = plt.scatter(param1_values, param2_values, c=objective_values, 
                                        cmap='viridis', alpha=0.6)
                    plt.colorbar(scatter, label=objective.replace('_', ' ').title())
                    plt.xlabel(param_names[0])
                    plt.ylabel(param_names[1])
                    plt.title(f'{objective.replace("_", " ").title()} vs Parameters')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'optimization_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Failed to create optimization plots: {e}")


def main():
    """Test the enhanced backtest engine"""
    # This would be implemented by a concrete strategy class
    pass


if __name__ == "__main__":
    main()
