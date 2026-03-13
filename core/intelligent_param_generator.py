"""
intelligent_param_generator.py

AI-powered parameter generator that analyzes factor characteristics
and generates intelligent parameter spaces (approximately 150 combinations).

Features:
- Factor data analysis (statistics, distribution, data type)
- LLM-based parameter space generation
- Adaptive parameter ranges based on factor characteristics
- Smart combination generation (~150 combinations)
"""

import json
import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import product
from datetime import datetime

from core.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FactorAnalyzer:
    """
    Analyzes factor data characteristics to determine optimal parameter ranges.
    """
    
    @staticmethod
    def analyze_factor_characteristics(factor_data: pd.DataFrame) -> Dict:
        """
        Analyze factor data to determine its characteristics.
        
        :param factor_data: DataFrame with 'timestamp' and 'value' columns
        :return: Dictionary with factor characteristics
        """
        if factor_data is None or len(factor_data) == 0:
            return {}
        
        values = factor_data['value'].dropna()
        
        if len(values) == 0:
            return {}
        
        # Basic statistics
        stats = {
            'count': len(values),
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'q25': float(values.quantile(0.25)),
            'q75': float(values.quantile(0.75)),
            'iqr': float(values.quantile(0.75) - values.quantile(0.25))
        }
        
        # Determine data type characteristics
        value_range = stats['max'] - stats['min']
        mean_abs = abs(stats['mean'])
        
        # Check if data is percentage-like (0-1 or 0-100 range)
        is_percentage = False
        if stats['min'] >= 0 and stats['max'] <= 1.5:
            is_percentage = True
        elif stats['min'] >= 0 and stats['max'] <= 150:
            # Could be 0-100 percentage
            if stats['max'] <= 105:
                is_percentage = True
        
        # Check if data is count-like (non-negative integers)
        is_count = False
        if stats['min'] >= 0 and (values % 1 == 0).all():
            is_count = True
        
        # Calculate percentage changes
        pct_changes = values.pct_change().dropna()
        abs_changes = values.diff().dropna()
        
        pct_stats = {}
        abs_stats = {}
        
        if len(pct_changes) > 0:
            pct_stats = {
                'pct_mean': float(pct_changes.mean()),
                'pct_std': float(pct_changes.std()),
                'pct_min': float(pct_changes.min()),
                'pct_max': float(pct_changes.max()),
                'pct_q25': float(pct_changes.quantile(0.25)),
                'pct_q75': float(pct_changes.quantile(0.75))
            }
        
        if len(abs_changes) > 0:
            abs_stats = {
                'abs_mean': float(abs_changes.mean()),
                'abs_std': float(abs_changes.std()),
                'abs_min': float(abs_changes.min()),
                'abs_max': float(abs_changes.max()),
                'abs_q25': float(abs_changes.quantile(0.25)),
                'abs_q75': float(abs_changes.quantile(0.75))
            }
        
        # Determine volatility
        if stats['mean'] != 0:
            cv = stats['std'] / abs(stats['mean'])  # Coefficient of variation
        else:
            cv = float('inf') if stats['std'] > 0 else 0
        
        # Detect volatility level
        if cv < 0.1:
            volatility_level = 'low'
        elif cv < 0.5:
            volatility_level = 'medium'
        else:
            volatility_level = 'high'
        
        # Detect typical change magnitude for parameter tuning
        typical_pct_change = abs(pct_stats.get('pct_mean', 0)) if pct_stats else 0
        typical_abs_change = abs(abs_stats.get('abs_mean', 0)) if abs_stats else 0
        
        # Determine if data is centered around specific value
        is_centered = False
        center_value = None
        if abs(stats['mean'] - stats['median']) < stats['std'] * 0.1:
            is_centered = True
            center_value = stats['mean']
        
        # Detect if data has specific scale (for threshold determination)
        scale_hint = None
        if is_percentage:
            if stats['max'] <= 1.5:
                scale_hint = '0-1_scale'
            else:
                scale_hint = '0-100_scale'
        elif stats['min'] >= 0:
            if stats['max'] < 10:
                scale_hint = 'small_positive'
            elif stats['max'] < 1000:
                scale_hint = 'medium_positive'
            else:
                scale_hint = 'large_positive'
        else:
            scale_hint = 'signed'
        
        characteristics = {
            'basic_stats': stats,
            'pct_change_stats': pct_stats,
            'abs_change_stats': abs_stats,
            'is_percentage': is_percentage,
            'is_count': is_count,
            'coefficient_of_variation': cv,
            'volatility_level': volatility_level,
            'value_range': value_range,
            'typical_pct_change': typical_pct_change,
            'typical_abs_change': typical_abs_change,
            'is_centered': is_centered,
            'center_value': center_value,
            'scale_hint': scale_hint,
            'data_type_hint': 'percentage' if is_percentage else ('count' if is_count else 'continuous')
        }
        
        return characteristics


class IntelligentParamGenerator:
    """
    Generates intelligent parameter spaces using LLM based on factor characteristics.
    """
    
    def __init__(self, llm_client: LLMClient, target_combinations: int = 500):
        """
        Initialize intelligent parameter generator.
        
        :param llm_client: LLM client for parameter generation
        :param target_combinations: Target number of parameter combinations (~500)
        """
        self.llm_client = llm_client
        self.target_combinations = target_combinations
        self.analyzer = FactorAnalyzer()
        
        logging.info(f"Intelligent Parameter Generator initialized")
        logging.info(f"  Target combinations: {target_combinations}")
    
    def analyze_and_generate_params(self,
                                   asset: str,
                                   factor_name: str,
                                   factor_data: pd.DataFrame,
                                   use_ai: bool = False) -> Dict:
        """
        Analyze factor and generate intelligent parameter space.

        Flow:
        1. classify_factor  - decide param_method from factor name + stats (LLM or heuristic)
        2. _build_grid_for_method - build long/short grid scaled to the chosen method
        3. AI-based rolling/param-space (optional, use_ai=True only)

        :param asset: Asset symbol
        :param factor_name: Factor name
        :param factor_data: Factor data DataFrame
        :param use_ai: Whether to use LLM for rolling/param-space refinement (default False)
        :return: Parameter grid dictionary
        """
        logging.info(f"Analyzing factor characteristics: {asset}/{factor_name}")

        if factor_data is None or len(factor_data) == 0:
            logging.warning("Empty factor data, using default params")
            return self._generate_default_params()

        # Step 0: Classify factor to decide param_method (LLM with heuristic fallback)
        param_method = self.classify_factor(factor_name, factor_data)
        logging.info(f"  Chosen param_method: {param_method}")

        # Step 0.5: Build method-aware grid (correct scale for long/short thresholds)
        param_grid = self._build_grid_for_method(param_method, factor_data)

        if not use_ai or not self.llm_client:
            param_grid = self._adjust_to_target_combinations(param_grid)
            logging.info(f"Generated {self._count_combinations(param_grid)} combinations using method-aware grid")
            return param_grid

        # Optional AI refinement of rolling windows and param ranges
        logging.info("Using AI-based parameter generation for refinement")
        characteristics = self.analyzer.analyze_factor_characteristics(factor_data)
        if not characteristics:
            logging.warning("Failed to analyze factor characteristics, keeping method-aware grid")
            return param_grid

        param_space = self._request_llm_param_generation(
            asset=asset,
            factor_name=factor_name,
            characteristics=characteristics
        )
        if param_space:
            ai_grid = self._generate_param_grid(param_space)
            # Keep param_method from classification; only adopt rolling/ranges from AI if valid
            if 'param_method' in ai_grid:
                ai_grid['param_method'] = [param_method]
            param_grid = self._adjust_to_target_combinations(ai_grid)
        else:
            logging.warning("LLM refinement failed, keeping method-aware grid")
            param_grid = self._adjust_to_target_combinations(param_grid)

        logging.info(f"Generated parameter grid: {self._count_combinations(param_grid)} combinations")
        return param_grid

    # ------------------------------------------------------------------
    # Step 0: Factor Classification
    # ------------------------------------------------------------------

    def classify_factor(self, factor_name: str, factor_data: pd.DataFrame) -> str:
        """
        Decide the best param_method for this factor.

        Priority order:
        1. LLM suggestion (if llm_client available) - based on name + stats summary only
        2. Heuristic fallback - deterministic rules on raw-value statistics

        Returns one of: 'pct_change', 'zscore', 'ratio_ma', 'minmax', 'raw', 'absolute'

        :param factor_name: Factor name string (used as semantic hint for LLM)
        :param factor_data: DataFrame with 'value' column
        :return: param_method string
        """
        VALID_METHODS = {'pct_change', 'zscore', 'ratio_ma', 'minmax', 'raw', 'absolute'}

        values = factor_data['value'].dropna() if factor_data is not None else pd.Series([], dtype=float)

        # Always try LLM first when available
        if self.llm_client and len(values) > 0:
            llm_method = self._classify_factor_via_llm(factor_name, values)
            if llm_method and llm_method in VALID_METHODS:
                logging.info(f"  LLM classification -> {llm_method}")
                return llm_method
            logging.info("  LLM classification unavailable or invalid, falling back to heuristic")

        return self._classify_factor_heuristic(values)

    def _classify_factor_via_llm(self, factor_name: str, values: pd.Series) -> Optional[str]:
        """
        Ask LLM to classify factor - sends only name + statistical summary, no backtest results.
        Returns param_method string or None on failure.
        """
        stats = {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'first_5': [round(float(v), 4) for v in values.head(5).tolist()],
            'last_5':  [round(float(v), 4) for v in values.tail(5).tolist()],
        }

        prompt = f"""You are a quant analyst. Given the factor name and its raw data statistics below,
decide the best normalization method for use in a backtesting signal.

Factor name: {factor_name}
Raw value stats:
  min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}
First 5 values: {stats['first_5']}
Last  5 values: {stats['last_5']}

Choose EXACTLY ONE method from this list and explain why IN ONE SENTENCE:
- pct_change : relative change (x_t - x_(t-1)) / x_(t-1). Good for SOPR-like ratios, bounded metrics.
- zscore     : (value - rolling_mean) / rolling_std. Best for level/trend data (addresses, balance, counts).
- ratio_ma   : value / rolling_mean. Good for volume, activity data. Output is a ratio around 1.0.
- minmax     : (value - rolling_min) / (rolling_max - rolling_min). Good for oscillators, bounded indicators.
- raw        : raw value directly. Only if already bounded (e.g. -1 to 1, 0 to 1).
- absolute   : diff(value). Rarely preferred; only for already-stationary level data.

DECISION RULE: Do NOT choose based on backtest results. Choose purely from the nature of the data.

Respond in JSON only:
{{"param_method": "<method>", "reason": "<one sentence>"}}"""

        messages = [
            {"role": "system", "content": "You are a quantitative analyst. Answer in JSON only, no extra text."},
            {"role": "user",   "content": prompt}
        ]

        try:
            response = self.llm_client._make_request(messages, temperature=0.1, max_tokens=200)
            if not response or not response.get('content'):
                return None

            text = response['content'].strip()
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                method = parsed.get('param_method', '').strip().lower()
                reason = parsed.get('reason', '')
                logging.info(f"  LLM reason: {reason}")
                return method
        except Exception as e:
            logging.warning(f"  LLM classification error: {e}")

        return None

    def _classify_factor_heuristic(self, values: pd.Series) -> str:
        """
        Heuristic fallback: decide param_method from raw value statistics alone.
        Uses the decision tree from the framework:
          - Bounded [-1.5, 1.5] or [0, 105]  → raw
          - Has upward trend (level data)      → zscore
          - Otherwise                          → pct_change (safe default)
        """
        if len(values) == 0:
            return 'pct_change'

        vmin = float(values.min())
        vmax = float(values.max())

        # Already bounded: sentiment scores, correlations, SOPR ~1
        if vmin >= -1.5 and vmax <= 1.5:
            return 'raw'
        if vmin >= 0 and vmax <= 105:
            return 'raw'

        # Detect long-term upward trend: compare first-quarter mean vs last-quarter mean
        n = len(values)
        q = max(1, n // 4)
        first_mean = float(values.iloc[:q].mean())
        last_mean  = float(values.iloc[-q:].mean())
        has_trend = (first_mean != 0) and (abs(last_mean / first_mean) > 1.2 or abs(last_mean / first_mean) < 0.83)

        if has_trend:
            return 'zscore'

        return 'pct_change'

    # ------------------------------------------------------------------
    # Step 0.5: Method-Aware Grid Building
    # ------------------------------------------------------------------

    def _build_grid_for_method(self, param_method: str, factor_data: pd.DataFrame) -> Dict:
        """
        Build param grid with long_param / short_param ranges correct for the chosen method.

        Each method produces rolling_return in a well-defined scale:
          pct_change  → ~ -0.1 to 0.1
          zscore      → ~ -3   to 3
          ratio_ma    → ~ 0.7  to 1.3
          minmax      → 0 to 1
          raw         → raw factor scale (uses rolling-mean percentiles)
          absolute    → diff scale (uses diff percentiles)

        :param param_method: One of the valid method strings
        :param factor_data: DataFrame with 'value' column
        :return: Parameter grid dict
        """
        rolling = [1, 3, 5, 7, 10, 14, 20, 30, 50]

        if param_method == 'zscore':
            long_param  = [-2.0, -1.5, -1.0, -0.5,  0.0]
            short_param = [ 0.0,  0.5,  1.0,  1.5,  2.0]

        elif param_method == 'ratio_ma':
            long_param  = [0.80, 0.85, 0.90, 0.95, 1.00]
            short_param = [1.00, 1.05, 1.10, 1.15, 1.20]

        elif param_method == 'minmax':
            long_param  = [0.10, 0.20, 0.30, 0.40, 0.50]
            short_param = [0.50, 0.60, 0.70, 0.80, 0.90]

        elif param_method in ('pct_change', 'absolute'):
            # Use pct_change stats for pct_change; diff stats for absolute
            values = factor_data['value'].dropna() if factor_data is not None else pd.Series([], dtype=float)
            if param_method == 'pct_change':
                series = values.pct_change().dropna()
            else:
                series = values.diff().dropna()

            if len(series) > 5:
                q10 = float(series.quantile(0.10))
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                q90 = float(series.quantile(0.90))
                mid = float(series.median())
                # long: signal triggers when factor is LOW (below threshold)
                long_param  = sorted(set([round(q10, 4), round(q25, 4), round(mid, 4)]))
                # short: signal triggers when factor is HIGH (above threshold)
                short_param = sorted(set([round(mid, 4), round(q75, 4), round(q90, 4)]))
            else:
                long_param  = [-0.05, -0.02, -0.01, 0.0,  0.01]
                short_param = [-0.01,  0.0,   0.01, 0.02, 0.05]

        elif param_method == 'raw':
            # Use rolling-mean distribution as signal; thresholds are percentiles of raw values
            values = factor_data['value'].dropna() if factor_data is not None else pd.Series([], dtype=float)
            if len(values) > 5:
                q10 = float(values.quantile(0.10))
                q25 = float(values.quantile(0.25))
                q75 = float(values.quantile(0.75))
                q90 = float(values.quantile(0.90))
                mid = float(values.median())
                long_param  = sorted(set([round(q10, 4), round(q25, 4), round(mid, 4)]))
                short_param = sorted(set([round(mid, 4), round(q75, 4), round(q90, 4)]))
            else:
                long_param  = [-0.05, -0.02, 0.0]
                short_param = [ 0.0,   0.02, 0.05]

        else:
            # Ultimate fallback: safe pct_change grid
            long_param  = [-0.05, -0.02, -0.01, 0.0,  0.01]
            short_param = [-0.01,  0.0,   0.01, 0.02, 0.05]

        return {
            'rolling':      rolling,
            'long_param':   long_param,
            'short_param':  short_param,
            'param_method': [param_method],
        }
    
    def _request_llm_param_generation(self,
                                     asset: str,
                                     factor_name: str,
                                     characteristics: Dict) -> Optional[Dict]:
        """
        Request LLM to generate parameter space based on factor characteristics.
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param characteristics: Factor characteristics dictionary
        :return: Parameter space dictionary or None if failed
        """
        logging.info(f"Requesting LLM to generate parameter space for {factor_name}")
        
        # Prepare prompt for LLM
        prompt = self._create_param_generation_prompt(factor_name, characteristics)
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional quantitative trading parameter optimization expert. Generate appropriate parameter space based on factor data characteristics. Please answer strictly in JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Request from LLM
        response = self.llm_client._make_request(messages, temperature=0.4, max_tokens=1500)
        
        if not response or not response.get('content'):
            logging.error("LLM parameter generation request failed")
            return None
        
        # Log full LLM response for debugging
        response_text = response['content']
        logging.info("=" * 80)
        logging.info("LLM Parameter Generation Response:")
        logging.info("=" * 80)
        logging.info(response_text)
        logging.info("=" * 80)
        
        # Parse LLM response
        param_space = self._parse_param_generation_response(response_text, characteristics)
        
        return param_space
    
    def _create_param_generation_prompt(self, factor_name: str, characteristics: Dict) -> str:
        """
        Create prompt for LLM parameter generation.
        
        :param factor_name: Factor name
        :param characteristics: Factor characteristics
        :return: Prompt string
        """
        basic_stats = characteristics.get('basic_stats', {})
        pct_stats = characteristics.get('pct_change_stats', {})
        abs_stats = characteristics.get('abs_change_stats', {})
        
        prompt = f"""Please generate appropriate parameter space based on the following factor data characteristics (target approximately 150 combinations).

Factor Name: {factor_name}

Data Characteristics:
- Data Type: {characteristics.get('data_type_hint', 'unknown')}
- Value Range: {basic_stats.get('min', 0):.4f} to {basic_stats.get('max', 0):.4f}
- Mean: {basic_stats.get('mean', 0):.4f}
- Standard Deviation: {basic_stats.get('std', 0):.4f}
- Median: {basic_stats.get('median', 0):.4f}
- Coefficient of Variation: {characteristics.get('coefficient_of_variation', 0):.4f}

Percentage Change Statistics:
- Mean Change: {pct_stats.get('pct_mean', 0):.6f}
- Change Std: {pct_stats.get('pct_std', 0):.6f}
- Change Range: {pct_stats.get('pct_min', 0):.6f} to {pct_stats.get('pct_max', 0):.6f}

Absolute Change Statistics:
- Mean Change: {abs_stats.get('abs_mean', 0):.4f}
- Change Std: {abs_stats.get('abs_std', 0):.4f}
- Change Range: {abs_stats.get('abs_min', 0):.4f} to {abs_stats.get('abs_max', 0):.4f}

Please answer in JSON format with the following fields:
{{
    "param_method": "pct_change" or "absolute" or "raw",
    "param_method_reason": "selection reason",
    "rolling_windows": [suggested rolling window list, e.g. [1, 3, 5, 10, 20]],
    "long_param_range": {{
        "method": "percentage" or "absolute",
        "min": minimum value,
        "max": maximum value,
        "step": step size,
        "reason": "range selection reason"
    }},
    "short_param_range": {{
        "method": "percentage" or "absolute",
        "min": minimum value,
        "max": maximum value,
        "step": step size,
        "reason": "range selection reason"
    }},
    "estimated_combinations": estimated number of combinations
}}

Notes:
1. If data is percentage type (0-1 or 0-100), prefer pct_change
2. If data is count type (non-negative integers), consider absolute
3. long_param is usually negative or smaller values (long signal)
4. short_param is usually positive or larger values (short signal)
5. Target is to generate approximately 150 parameter combinations

Answer only in JSON, no other text."""
        
        return prompt
    
    def _parse_param_generation_response(self, response_text: str, characteristics: Dict) -> Optional[Dict]:
        """
        Parse LLM response and extract parameter space.
        
        :param response_text: LLM response text
        :param characteristics: Factor characteristics for validation
        :return: Parameter space dictionary or None if failed
        """
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                param_space = json.loads(json_str)
                
                # Validate and normalize
                param_space = self._validate_param_space(param_space, characteristics)
                
                return param_space
            else:
                logging.warning("No JSON found in LLM response")
                return None
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            logging.error(f"Response text: {response_text[:200]}...")
            return None
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return None
    
    def _validate_param_space(self, param_space: Dict, characteristics: Dict) -> Dict:
        """
        Validate and normalize parameter space.
        
        :param param_space: Parameter space from LLM
        :param characteristics: Factor characteristics
        :return: Validated parameter space
        """
        # Ensure required fields
        if 'param_method' not in param_space:
            param_space['param_method'] = 'pct_change'
        
        if 'rolling_windows' not in param_space:
            param_space['rolling_windows'] = [1, 5, 10, 20]
        
        # Validate rolling windows
        rolling_windows = param_space['rolling_windows']
        if not isinstance(rolling_windows, list) or len(rolling_windows) == 0:
            param_space['rolling_windows'] = [1, 5, 10, 20]
        else:
            # Ensure positive integers
            param_space['rolling_windows'] = [max(1, int(r)) for r in rolling_windows if r > 0]
        
        # Validate long_param_range
        if 'long_param_range' not in param_space:
            param_space['long_param_range'] = {
                'method': 'percentage',
                'min': -0.2,
                'max': 0.0,
                'step': 0.02
            }
        
        # Validate short_param_range
        if 'short_param_range' not in param_space:
            param_space['short_param_range'] = {
                'method': 'percentage',
                'min': 0.0,
                'max': 0.2,
                'step': 0.02
            }
        
        return param_space
    
    def _generate_rule_based_params(self, characteristics: Dict) -> Dict:
        """
        Generate parameter space using rule-based approach when LLM fails.
        Adapts parameter ranges based on data characteristics to avoid missing important combinations.
        
        :param characteristics: Factor characteristics
        :return: Parameter space dictionary
        """
        basic_stats = characteristics.get('basic_stats', {})
        pct_stats = characteristics.get('pct_change_stats', {})
        abs_stats = characteristics.get('abs_change_stats', {})
        volatility_level = characteristics.get('volatility_level', 'medium')
        scale_hint = characteristics.get('scale_hint', 'signed')
        typical_pct_change = characteristics.get('typical_pct_change', 0.01)
        typical_abs_change = characteristics.get('typical_abs_change', 0)
        
        # Determine param_method based on data type
        if characteristics.get('is_percentage', False):
            param_method = 'pct_change'
        elif characteristics.get('is_count', False):
            param_method = 'absolute'
        else:
            # Use coefficient of variation to decide
            cv = characteristics.get('coefficient_of_variation', 1.0)
            if cv > 1.0:
                param_method = 'pct_change'  # High volatility, use percentage
            else:
                param_method = 'absolute'  # Low volatility, use absolute
        
        # Generate rolling windows based on volatility and data characteristics
        volatility_level = characteristics.get('volatility_level', 'medium')
        if volatility_level == 'low':
            # Low volatility: use longer windows to smooth out noise
            rolling_windows = [5, 10, 14, 20, 30, 50]
        elif volatility_level == 'high':
            # High volatility: use shorter windows to catch quick changes
            rolling_windows = [1, 3, 5, 7, 10, 14]
        else:
            # Medium volatility: balanced approach
            rolling_windows = [1, 3, 5, 7, 10, 14, 20, 30]
        
        # Generate long/short param ranges based on statistics and data characteristics
        if param_method == 'pct_change':
            # Use percentage change statistics
            pct_std = abs(pct_stats.get('pct_std', 0.05))
            pct_mean = pct_stats.get('pct_mean', 0)
            
            # Full symmetric range around mean, no directional assumption
            # long_param: lower half; short_param: upper half
            spread = 3 * pct_std if pct_std > 0 else 3 * typical_pct_change if typical_pct_change > 0 else 0.1
            long_min = pct_mean - spread
            long_max = pct_mean
            short_min = pct_mean
            short_max = pct_mean + spread
            
            # Adjust step size based on volatility
            if volatility_level == 'low':
                step_base = max(0.005, min(pct_std / 10, typical_pct_change / 5))
            elif volatility_level == 'high':
                step_base = max(0.01, min(pct_std / 3, typical_pct_change / 2))
            else:
                step_base = max(0.01, min(pct_std / 5, typical_pct_change / 3))
            
            long_step = step_base
            short_step = step_base
            
            long_param_range = {
                'method': 'percentage',
                'min': long_min,
                'max': long_max,
                'step': long_step
            }
            
            short_param_range = {
                'method': 'percentage',
                'min': short_min,
                'max': short_max,
                'step': short_step
            }
        else:
            # Use absolute change statistics
            abs_std = abs(abs_stats.get('abs_std', 100))
            abs_mean = abs_stats.get('abs_mean', 0)
            
            # Full symmetric range around mean, no directional assumption
            # long_param: lower half; short_param: upper half
            # Both can be negative, both positive, or mixed
            range_low = abs_mean - 3 * abs_std
            range_high = abs_mean + 3 * abs_std
            midpoint = abs_mean

            long_min = range_low
            long_max = midpoint
            short_min = midpoint
            short_max = range_high
            
            # Adjust step based on scale
            if scale_hint == 'small_positive':
                step_base = max(0.1, typical_abs_change / 5) if typical_abs_change > 0 else abs_std / 10
            elif scale_hint == 'medium_positive':
                step_base = max(1, typical_abs_change / 5) if typical_abs_change > 0 else abs_std / 10
            else:
                step_base = max(abs_std / 10, 10)
            
            long_step = step_base
            short_step = step_base
            
            long_param_range = {
                'method': 'absolute',
                'min': long_min,
                'max': long_max,
                'step': long_step
            }
            
            short_param_range = {
                'method': 'absolute',
                'min': short_min,
                'max': short_max,
                'step': short_step
            }
        
        return {
            'param_method': param_method,
            'rolling_windows': rolling_windows,
            'long_param_range': long_param_range,
            'short_param_range': short_param_range
        }
    
    def _generate_param_grid(self, param_space: Dict) -> Dict:
        """
        Generate parameter grid from parameter space.
        
        :param param_space: Parameter space dictionary
        :return: Parameter grid dictionary
        """
        # Generate rolling windows
        rolling = param_space.get('rolling_windows', [1, 5, 10, 20])
        
        # Generate long params
        long_range = param_space.get('long_param_range', {})
        long_params = self._generate_param_list(long_range)
        
        # Generate short params
        short_range = param_space.get('short_param_range', {})
        short_params = self._generate_param_list(short_range)
        
        return {
            'rolling': rolling,
            'long_param': long_params,
            'short_param': short_params,
            'param_method': [param_space.get('param_method', 'pct_change')]
        }
    
    def _generate_param_list(self, param_range: Dict) -> List[float]:
        """
        Generate parameter list from range specification.
        
        :param param_range: Range dictionary with min, max, step
        :return: List of parameter values
        """
        min_val = param_range.get('min', -0.2)
        max_val = param_range.get('max', 0.2)
        step = param_range.get('step', 0.02)
        
        if step <= 0:
            step = 0.02
        
        # Generate list
        param_list = []
        current = min_val
        
        while current <= max_val:
            param_list.append(round(current, 6))
            current += step
        
        # Ensure max is included
        if param_list[-1] < max_val:
            param_list.append(round(max_val, 6))
        
        return param_list
    
    def _count_combinations(self, param_grid: Dict) -> int:
        """
        Count total parameter combinations.
        
        :param param_grid: Parameter grid dictionary
        :return: Number of combinations
        """
        from itertools import product
        
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        return len(combinations)
    
    def _adjust_to_target_combinations(self, param_grid: Dict) -> Dict:
        """
        Adjust parameter grid to target approximately 150 combinations.
        
        :param param_grid: Parameter grid dictionary
        :return: Adjusted parameter grid
        """
        current_combinations = self._count_combinations(param_grid)
        target = self.target_combinations
        
        if current_combinations <= target * 1.2:
            # Close enough, return as is
            return param_grid
        
        # Need to reduce combinations
        logging.info(f"Adjusting from {current_combinations} to ~{target} combinations")
        
        # Calculate reduction factor
        reduction_factor = (target / current_combinations) ** (1/3)  # Cube root for 3D grid
        
        # Reduce each dimension
        adjusted_grid = param_grid.copy()
        
        # Reduce rolling windows
        rolling = adjusted_grid.get('rolling', [])
        if len(rolling) > 0:
            new_rolling_count = max(2, int(len(rolling) * reduction_factor))
            step = max(1, len(rolling) // new_rolling_count)
            adjusted_grid['rolling'] = rolling[::step][:new_rolling_count]
        
        # Reduce long params
        long_params = adjusted_grid.get('long_param', [])
        if len(long_params) > 0:
            new_long_count = max(3, int(len(long_params) * reduction_factor))
            step = max(1, len(long_params) // new_long_count)
            adjusted_grid['long_param'] = long_params[::step][:new_long_count]
        
        # Reduce short params
        short_params = adjusted_grid.get('short_param', [])
        if len(short_params) > 0:
            new_short_count = max(3, int(len(short_params) * reduction_factor))
            step = max(1, len(short_params) // new_short_count)
            adjusted_grid['short_param'] = short_params[::step][:new_short_count]
        
        final_combinations = self._count_combinations(adjusted_grid)
        logging.info(f"Adjusted to {final_combinations} combinations")
        
        return adjusted_grid
    
    def _generate_params_from_min_max(self, factor_data: pd.DataFrame) -> Dict:
        """
        Generate parameter grid based on min/max values of factor data.
        Simple, fast, and reliable approach.
        
        :param factor_data: Factor data DataFrame with 'value' column
        :return: Parameter grid dictionary
        """
        if factor_data is None or len(factor_data) == 0:
            logging.warning("Empty factor data, using default params")
            return self._generate_default_params()
        
        values = factor_data['value'].dropna()
        
        if len(values) == 0:
            logging.warning("No valid values, using default params")
            return self._generate_default_params()
        
        # Calculate statistics
        data_min = float(values.min())
        data_max = float(values.max())
        data_mean = float(values.mean())
        data_std = float(values.std())
        
        # Fixed rolling window (simplified)
        rolling = [1, 3, 5, 7, 10, 14, 20, 30, 50]
        
        # Determine param_method based on data characteristics
        value_range = data_max - data_min
        is_percentage = (data_min >= 0 and data_max <= 1.5) or (data_min >= 0 and data_max <= 105)
        
        if is_percentage:
            param_method = 'pct_change'
        else:
            # Use coefficient of variation to decide
            cv = data_std / abs(data_mean) if data_mean != 0 else float('inf')
            param_method = 'pct_change' if cv > 1.0 else 'absolute'
        
        # Generate parameter ranges based on data range
        # Use mean ± 3*std to avoid extreme outliers
        if param_method == 'pct_change':
            # For percentage-based: use reasonable range around mean
            reasonable_min = max(data_min, data_mean - 3 * data_std)
            reasonable_max = min(data_max, data_mean + 3 * data_std)
            
            # Ensure we have negative and positive ranges
            long_max = min(0, reasonable_max) if reasonable_max < 0 else -0.01
            long_min = max(reasonable_min, -0.5)  # Cap at -50%
            
            short_min = max(0, reasonable_min) if reasonable_min > 0 else 0.01
            short_max = min(reasonable_max, 0.5)  # Cap at +50%
            
            # Generate evenly spaced values
            # Target: ~10 values each for long and short
            # Total combinations: 9 (rolling) × 10 (long) × 10 (short) = 900
            num_params = 10
            
            if long_min < long_max:
                long_param = np.linspace(long_min, long_max, num_params).tolist()
                long_param = [round(x, 4) for x in long_param]
            else:
                # Fallback to default range
                long_param = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.08, -0.05, -0.03, -0.02, -0.01]
            
            if short_min < short_max:
                short_param = np.linspace(short_min, short_max, num_params).tolist()
                short_param = [round(x, 4) for x in short_param]
            else:
                # Fallback to default range
                short_param = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
        else:
            # For absolute-based: rolling_return = diff(factor).rolling(window).mean()
            # Use diff statistics to match the actual signal distribution
            diff_values = values.diff().dropna()
            diff_std = float(diff_values.std()) if len(diff_values) > 1 else data_std * 0.01
            diff_mean = float(diff_values.mean()) if len(diff_values) > 1 else 0.0

            # Full symmetric range: mean ± 3*std, no directional assumption
            range_low = diff_mean - 3 * diff_std
            range_high = diff_mean + 3 * diff_std
            midpoint = diff_mean

            num_params = 10

            # long_param: lower half of the range (below midpoint)
            # short_param: upper half of the range (above midpoint)
            # This guarantees long_param < short_param (required for signal logic)
            # but allows both to be negative, both positive, or mixed
            long_param = np.linspace(range_low, midpoint, num_params + 1)[:-1].tolist()
            long_param = [round(x, 4) for x in long_param]

            short_param = np.linspace(midpoint, range_high, num_params + 1)[1:].tolist()
            short_param = [round(x, 4) for x in short_param]
        
        param_grid = {
            'rolling': rolling,
            'long_param': long_param,
            'short_param': short_param,
            'param_method': [param_method]
        }
        
        num_combinations = self._count_combinations(param_grid)
        logging.info(f"Generated min/max based grid: {num_combinations} combinations")
        logging.info(f"  Rolling: {len(rolling)} options")
        logging.info(f"  Long param: {len(long_param)} options (range: {long_param[0]:.4f} to {long_param[-1]:.4f})")
        logging.info(f"  Short param: {len(short_param)} options (range: {short_param[0]:.4f} to {short_param[-1]:.4f})")
        
        return param_grid
    
    def _generate_default_params(self) -> Dict:
        """
        Generate default parameter space when analysis fails.
        
        :return: Default parameter grid
        """
        return {
            'rolling': [1, 3, 5, 7, 10, 14, 20, 30, 50],
            'long_param': [-0.3, -0.25, -0.2, -0.15, -0.1, -0.08, -0.05, -0.03, -0.02, -0.01],
            'short_param': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3],
            'param_method': ['pct_change']
        }
