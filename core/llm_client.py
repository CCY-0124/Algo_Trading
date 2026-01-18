"""
llm_client.py

LLM client for communicating with Ollama service on Jetson Orin Nano.
Handles HTTP requests, error handling, retry mechanisms, and structured prompts.

Features:
- HTTP communication with Ollama API
- Error handling and retry mechanism
- Structured prompt generation
- Batch analysis support
- Timeout handling
- Response parsing and validation
"""

import requests
import json
import logging
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LLMClient:
    """
    Client for communicating with Ollama LLM service on Jetson.
    
    Supports:
    - Single factor analysis
    - Batch factor analysis
    - Structured prompt generation
    - Error handling and retries
    - Response parsing
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:11434/api/chat",
                 model: str = "qwen2.5:3b",
                 timeout: int = 120,
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize LLM client.
        
        :param api_url: Ollama API endpoint URL
        :param model: Model name (default: qwen2.5:3b)
        :param timeout: Request timeout in seconds
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logging.info(f"LLM Client initialized")
        logging.info(f"  API URL: {api_url}")
        logging.info(f"  Model: {model}")
        logging.info(f"  Timeout: {timeout}s")
    
    def _make_request(self, messages: List[Dict], temperature: float = 0.3, 
                     max_tokens: int = 2000) -> Optional[Dict]:
        """
        Make HTTP request to Ollama API with retry mechanism.
        
        :param messages: List of message dicts with 'role' and 'content'
        :param temperature: Sampling temperature
        :param max_tokens: Maximum tokens to generate
        :return: Response dictionary or None if failed
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            try:
                logging.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract content from Ollama response
                    content = result.get('message', {}).get('content', '')
                    if not content:
                        # Sometimes Ollama returns content directly
                        content = result.get('response', '')
                    
                    return {
                        'content': content,
                        'model': result.get('model', self.model),
                        'done': result.get('done', True)
                    }
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logging.error(f"LLM API error: {error_msg}")
                    
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logging.info(f"Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        return None
                        
            except requests.exceptions.Timeout:
                logging.error(f"Request timeout after {self.timeout}s")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logging.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    return None
                    
            except requests.exceptions.ConnectionError:
                logging.error(f"Connection error: Cannot reach {self.api_url}")
                logging.error("Please ensure Ollama is running on Jetson")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logging.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    return None
                    
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def _create_structured_prompt(self, factor_data: Dict) -> str:
        """
        Create structured prompt for factor analysis.
        
        :param factor_data: Dictionary containing factor information
        :return: Formatted prompt string
        """
        factor_name = factor_data.get('factor_name', 'Unknown')
        backtest_results = factor_data.get('backtest_results', {})
        factor_params = factor_data.get('factor_params', {})
        historical_context = factor_data.get('historical_context', '')
        
        prompt = f"""You are a quantitative trading analysis expert. Please answer strictly in the following format.

Factor Analysis Task:
Factor Name: {factor_name}
Parameter Settings: {json.dumps(factor_params, indent=2, ensure_ascii=False)}
Backtest Results:
- Total Return: {backtest_results.get('total_return', 0):.2%}
- Annual Return: {backtest_results.get('annual_return', 0):.2%}
- Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.4f}
- Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}
- Win Rate: {backtest_results.get('win_rate', 0):.2%}
- Number of Trades: {backtest_results.get('num_trades', 0)}
- Data Points: {backtest_results.get('data_points', 0)}

Historical Analysis Records: {historical_context if historical_context else 'No historical records'}

Please analyze and answer in JSON format:
{{
    "has_potential": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief reason (max 50 characters)",
    "buy_condition_suggestion": {{
        "method": "percentage" or "absolute",
        "threshold": value,
        "reason": "reason"
    }},
    "sell_condition_suggestion": {{
        "method": "percentage" or "absolute",
        "threshold": value,
        "reason": "reason"
    }},
    "rolling_window_suggestion": {{
        "range": [min, max],
        "recommended": recommended value,
        "reason": "reason"
    }},
    "optimization_priority": ["parameter1", "parameter2", "parameter3"],
    "risk_warning": "risk warning (if any)"
}}

Answer only in JSON, no other text."""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse LLM response and extract JSON.
        
        :param response_text: Raw response text from LLM
        :return: Parsed dictionary or fallback result
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['has_potential', 'confidence', 'reason']
                if all(field in parsed for field in required_fields):
                    return parsed
                else:
                    logging.warning("LLM response missing required fields")
                    return self._create_fallback_response(response_text)
            else:
                logging.warning("No JSON found in LLM response")
                return self._create_fallback_response(response_text)
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            logging.error(f"Response text: {response_text[:200]}...")
            return self._create_fallback_response(response_text)
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return self._create_fallback_response(response_text)
    
    def _create_fallback_response(self, response_text: str) -> Dict:
        """
        Create fallback response when parsing fails.
        
        :param response_text: Original response text
        :return: Fallback dictionary
        """
        # Try to extract basic information from text
        has_potential = any(word in response_text.lower() 
                          for word in ['good', 'promising', 'positive', 'potential', 'worth'])
        
        return {
            "has_potential": has_potential,
            "confidence": 0.3,  # Low confidence for fallback
            "reason": "LLM response parsing failed, using fallback analysis",
            "buy_condition_suggestion": {
                "method": "percentage",
                "threshold": 0.0,
                "reason": "Fallback: No specific suggestion"
            },
            "sell_condition_suggestion": {
                "method": "percentage",
                "threshold": 0.0,
                "reason": "Fallback: No specific suggestion"
            },
            "rolling_window_suggestion": {
                "range": [1, 20],
                "recommended": 10,
                "reason": "Fallback: Default range"
            },
            "optimization_priority": ["rolling", "long_param", "short_param"],
            "risk_warning": "LLM response parsing failed, results may be inaccurate",
            "raw_response": response_text[:500]  # Store first 500 chars for debugging
        }
    
    def analyze_factor(self, factor_data: Dict) -> Optional[Dict]:
        """
        Analyze a single factor using LLM.
        
        :param factor_data: Dictionary containing:
            - factor_name: str
            - backtest_results: dict
            - factor_params: dict
            - historical_context: str (optional)
        :return: Analysis result dictionary or None if failed
        """
        logging.info(f"Analyzing factor: {factor_data.get('factor_name', 'Unknown')}")
        
        # Create structured prompt
        prompt = self._create_structured_prompt(factor_data)
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are a professional quantitative trading analysis expert, skilled in analyzing trading strategies and parameter optimization. Please answer strictly in the required JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make request
        response = self._make_request(messages, temperature=0.3, max_tokens=2000)
        
        if response and response.get('content'):
            # Parse response
            analysis_result = self._parse_llm_response(response['content'])
            
            # Add metadata
            analysis_result['timestamp'] = datetime.now().isoformat()
            analysis_result['model'] = response.get('model', self.model)
            analysis_result['factor_name'] = factor_data.get('factor_name', 'Unknown')
            
            logging.info(f"Analysis completed for {factor_data.get('factor_name', 'Unknown')}")
            return analysis_result
        else:
            logging.error(f"Failed to get LLM response for {factor_data.get('factor_name', 'Unknown')}")
            return None
    
    def analyze_factor_iterative(self, factor_data: Dict) -> Optional[Dict]:
        """
        Analyze factor with full data for iterative optimization.
        
        This method sends complete factor data to LLM for better analysis.
        Used in iterative optimization where LLM can see the full data pattern.
        
        :param factor_data: Dictionary containing:
            - factor_name: str
            - current_params: dict (current parameters being tested)
            - current_result: dict (current backtest results)
            - iteration: int (current iteration number)
            - previous_iterations: list (previous iteration results)
            - top_stage1_results: list (top results from Stage 1)
            - factor_data: list (complete factor data as list of dicts)
            - data_summary: dict (data statistics if data is sampled)
            - historical_context: str (optional)
        :return: Analysis result dictionary or None if failed
        """
        factor_name = factor_data.get('factor_name', 'Unknown')
        iteration = factor_data.get('iteration', 1)
        
        logging.info(f"Analyzing factor iteratively: {factor_name} (iteration {iteration})")
        
        # Create prompt with full data
        prompt = self._create_iterative_prompt(factor_data)
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are a professional quantitative trading analysis expert, skilled in analyzing trading strategies and parameter optimization. You can now see complete factor data. Please perform deep analysis based on data patterns and historical results. Please answer strictly in the required JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make request with higher temperature for more creative suggestions
        response = self._make_request(messages, temperature=0.4, max_tokens=2500)
        
        if response and response.get('content'):
            # Parse response
            analysis_result = self._parse_llm_response(response['content'])
            
            # Add metadata
            analysis_result['timestamp'] = datetime.now().isoformat()
            analysis_result['model'] = response.get('model', self.model)
            analysis_result['factor_name'] = factor_name
            analysis_result['iteration'] = iteration
            
            logging.info(f"Iterative analysis completed for {factor_name} (iteration {iteration})")
            return analysis_result
        else:
            logging.error(f"Failed to get LLM response for {factor_name} (iteration {iteration})")
            return None
    
    def _create_iterative_prompt(self, factor_data: Dict) -> str:
        """
        Create prompt for iterative optimization with full data.
        
        :param factor_data: Dictionary containing factor information and data
        :return: Formatted prompt string
        """
        import json
        
        factor_name = factor_data.get('factor_name', 'Unknown')
        current_params = factor_data.get('current_params', {})
        current_result = factor_data.get('current_result', {})
        iteration = factor_data.get('iteration', 1)
        previous_iterations = factor_data.get('previous_iterations', [])
        top_stage1_results = factor_data.get('top_stage1_results', [])
        historical_context = factor_data.get('historical_context', '')
        raw_factor_data = factor_data.get('factor_data', [])
        data_summary = factor_data.get('data_summary', {})
        
        prompt = f"""You are a quantitative trading analysis expert conducting iterative optimization. Please perform deep analysis based on complete factor data and historical results.

Factor Name: {factor_name}
Current Iteration: {iteration}

Current Parameter Settings:
{json.dumps(current_params, indent=2, ensure_ascii=False)}

Current Backtest Results:
- Sharpe Ratio: {current_result.get('sharpe_ratio', 0):.4f}
- Calmar Ratio: {current_result.get('calmar_ratio', 0):.4f}
- Total Return: {current_result.get('total_return', 0):.2%}
- Annual Return: {current_result.get('annual_return', 0):.2%}
- Max Drawdown: {current_result.get('max_drawdown', 0):.2%}
- Win Rate: {current_result.get('win_rate', 0):.2%}
- Number of Trades: {current_result.get('num_trades', 0)}
"""
        
        # Add previous iterations context
        if previous_iterations:
            prompt += "\nPrevious Iteration Results:\n"
            for prev_iter in previous_iterations:
                prev_result = prev_iter.get('result', {})
                prompt += f"- Iteration {prev_iter.get('iteration', '?')}: "
                prompt += f"Sharpe {prev_result.get('sharpe_ratio', 0):.4f}, "
                prompt += f"Params {prev_iter.get('params', {})}\n"
        
        # Add top Stage 1 results for context
        if top_stage1_results:
            prompt += "\nStage 1 Best Results (for reference):\n"
            for i, result in enumerate(top_stage1_results[:3], 1):
                prompt += f"- Rank {i}: Sharpe {result.get('sharpe_ratio', 0):.4f}, "
                prompt += f"Params {result.get('params', {})}\n"
        
        # Add historical context
        if historical_context:
            prompt += f"\nHistorical Analysis Records:\n{historical_context}\n"
        
        # Add complete factor data
        if raw_factor_data:
            if data_summary:
                prompt += f"\nFactor Data Summary:\n"
                prompt += f"- Total Data Points: {data_summary.get('total_rows', len(raw_factor_data))}\n"
                if data_summary.get('date_range'):
                    prompt += f"- Date Range: {data_summary['date_range'].get('start')} to {data_summary['date_range'].get('end')}\n"
            
            prompt += f"\nComplete Factor Data (Total {len(raw_factor_data)} rows, all data):\n"
            # Format data as JSON array for better readability
            # Send all data rows
            prompt += json.dumps(raw_factor_data, indent=2, ensure_ascii=False)
        
        prompt += """
Please analyze the patterns, trends, and characteristics of the factor data based on the complete information above, then provide optimization suggestions.

Please answer in JSON format:
{
    "has_potential": true/false,
    "confidence": 0.0-1.0,
    "reason": "analysis reason (based on data patterns)",
    "data_insights": "data characteristics you observed (trends, cycles, anomalies, etc.)",
    "buy_condition_suggestion": {
        "method": "percentage" or "absolute",
        "threshold": value,
        "reason": "reason based on data analysis"
    },
    "sell_condition_suggestion": {
        "method": "percentage" or "absolute",
        "threshold": value,
        "reason": "reason based on data analysis"
    },
    "rolling_window_suggestion": {
        "range": [min, max],
        "recommended": recommended value,
        "reason": "reason based on data cycles and volatility"
    },
    "optimization_priority": ["parameter1", "parameter2", "parameter3"],
    "risk_warning": "risk warning (if any)",
    "next_iteration_focus": "what to focus on in the next iteration"
}

Answer only in JSON, no other text."""
        
        return prompt
    
    def analyze_batch(self, factors_data: List[Dict]) -> List[Dict]:
        """
        Analyze multiple factors in batch.
        
        :param factors_data: List of factor data dictionaries
        :return: List of analysis results
        """
        logging.info(f"Analyzing batch of {len(factors_data)} factors")
        
        results = []
        for i, factor_data in enumerate(factors_data, 1):
            logging.info(f"Processing factor {i}/{len(factors_data)}: {factor_data.get('factor_name', 'Unknown')}")
            
            result = self.analyze_factor(factor_data)
            if result:
                results.append(result)
            else:
                # Create fallback result
                fallback = {
                    "factor_name": factor_data.get('factor_name', 'Unknown'),
                    "has_potential": False,
                    "confidence": 0.0,
                    "reason": "LLM analysis failed",
                    "timestamp": datetime.now().isoformat()
                }
                results.append(fallback)
            
            # Small delay between requests to avoid overwhelming the server
            if i < len(factors_data):
                time.sleep(0.5)
        
        logging.info(f"Batch analysis completed: {len(results)}/{len(factors_data)} successful")
        return results
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama service.
        
        :return: True if connection successful, False otherwise
        """
        logging.info("Testing connection to Ollama service...")
        
        test_messages = [
            {
                "role": "user",
                "content": "Hello, this is a connection test. Please respond with 'OK'."
            }
        ]
        
        response = self._make_request(test_messages, temperature=0.1, max_tokens=10)
        
        if response and response.get('content'):
            logging.info("[OK] Connection test successful")
            logging.info(f"Response: {response['content'][:100]}")
            return True
        else:
            logging.error("[ERROR] Connection test failed")
            return False


if __name__ == "__main__":
    # Test the LLM client
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        jetson_ip = sys.argv[1]
        api_url = f"http://{jetson_ip}:11434/api/chat"
    else:
        api_url = "http://localhost:11434/api/chat"
    
    client = LLMClient(api_url=api_url)
    
    # Test connection
    if client.test_connection():
        print("\n✅ LLM Client is working correctly!")
        
        # Test factor analysis
        test_factor_data = {
            "factor_name": "BTC_sopr",
            "backtest_results": {
                "total_return": 0.25,
                "annual_return": 0.20,
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "num_trades": 45,
                "data_points": 1000
            },
            "factor_params": {
                "rolling": 10,
                "long_param": -0.04,
                "short_param": 0.14,
                "param_method": "pct_change"
            },
            "historical_context": ""
        }
        
        print("\nTesting factor analysis...")
        result = client.analyze_factor(test_factor_data)
        
        if result:
            print("\n✅ Analysis result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n❌ Analysis failed")
    else:
        print("\n❌ Cannot connect to Ollama service")
        print("Please ensure:")
        print("1. Ollama is running on Jetson")
        print("2. Qwen 2.5 3B model is downloaded: ollama pull qwen2.5:3b")
        print("3. Network connection is available")
        print("4. Firewall allows connection to port 11434")
