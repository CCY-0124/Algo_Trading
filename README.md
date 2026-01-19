# Algorithmic Trading System

A comprehensive algorithmic trading platform for cryptocurrency markets, featuring real-time data collection, backtesting, strategy optimization, and live trading capabilities.

## Features

### Core Functionality
- **Real-time Data Collection** - Automated Glassnode API data downloading with rate limiting and error handling
- **Advanced Backtesting Engine** - High-performance backtesting with multiple performance metrics
- **Strategy Optimization** - Parameter optimization with sensitivity analysis and structured output
- **Live Trading** - Real-time trading execution on Bybit exchange
- **Data Management** - Comprehensive data validation, cleaning, and storage with intelligent CSV processing
- **LLM-Powered Analysis** - Intelligent factor screening and parameter optimization using LLM

### Data Sources
- **Glassnode API** - On-chain and market data for cryptocurrencies
- **Multiple API Keys** - Automatic key rotation and fallback mechanisms
- **Local Data Storage** - Efficient CSV-based data storage with incremental updates

### Trading Strategies
- **Non-Price Strategy** - Advanced strategy using on-chain metrics and market indicators
- **Modular Architecture** - Easy to extend with new strategies
- **Parameter Optimization** - Automated strategy parameter tuning with user confirmation

## Project Structure

```
Algo_Trading/
├── config/                        # Configuration and settings
│   ├── __init__.py               # Config module entry
│   ├── paths.py                  # Path configuration
│   ├── trading_config.py         # Trading configuration
│   ├── download_config.py        # Download rate limiting configuration
│   └── secrets.py                # Encrypted API key management
├── core/                         # Core engine components
│   ├── __init__.py               # Core module entry
│   ├── enhanced_engine.py        # Enhanced backtesting engine
│   ├── llm_client.py             # LLM client for intelligent analysis
│   ├── llm_scheduler.py          # Intelligent LLM task scheduling
│   ├── context_manager.py        # Context management for LLM
│   ├── factor_screening.py       # Two-stage factor screening
│   ├── intelligent_param_generator.py  # LLM-powered parameter generation
│   ├── performance_monitor.py    # Performance monitoring
│   ├── data_cache.py             # Lightweight data caching
│   └── factor_status_tracker.py  # Factor status tracking
├── scripts/                      # Executable scripts
│   ├── daily_download.py         # Daily data downloader
│   ├── interactive_trading.py    # Interactive trading interface
│   ├── llm_backtest_agent.py     # Automated LLM backtest agent
│   ├── llm_orchestrator.py       # LLM workflow orchestrator
│   ├── schedule_daily_agent.py  # Windows task scheduler helper
│   └── setup_environment.py     # Environment setup
├── tools/                        # One-time setup and utility scripts
│   ├── generate_metadata_catalog.py  # Metadata catalog generation
│   ├── legacy_algorithm.py       # Legacy algorithm (for verification)
│   ├── simple_price_check.py    # API connectivity test
│   └── simple_optimization.py   # Quick parameter testing
├── strategies/                   # Trading strategies
│   ├── __init__.py               # Strategies module entry
│   └── enhanced_non_price_strategy.py  # Enhanced non-price strategy
├── trading/                      # Live trading components
│   ├── brokers/                  # Exchange connectivity
│   ├── execution/                # Order execution
│   └── real_time/                # Real-time trading
├── utils/                        # Utility functions
│   ├── local_data_loader.py      # Intelligent local data management
│   ├── strategy_manager.py       # Strategy management
│   ├── log_config.py             # Logging configuration
│   └── report_generator.py       # Report generation
├── optimization_reports/         # Optimization results (structured)
└── notebooks/                    # Jupyter notebooks
```

## Installation

### Prerequisites
- Python 3.9+ (Python 3.11+ recommended, Python 3.13 supported)
- Glassnode API key(s)
- Bybit API credentials (for live trading)
- LLM API access (for LLM features, optional)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Algo_Trading
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env file with your API keys
# GLASSNODE_API_KEY=your_glassnode_key
# GLASSNODE_API_KEY_BACKUP=your_backup_key
# BYBIT_API_KEY=your_bybit_key
# BYBIT_SECRET=your_bybit_secret
```

4. **Verify setup**
```bash
python tools/simple_price_check.py
```

## Data Management

### Intelligent Data Loading
The system includes an intelligent data loader with the following features:

- **Smart CSV Processing** - Handles both ('t', 'v') and ('timestamp', 'value') formats
- **Data Validation** - Validates timestamps, numeric values, and data integrity
- **Automatic Standardization** - Converts all data to ('timestamp', 'value') format
- **Date Overlap Checking** - Ensures price and factor data have overlapping dates
- **Multi-Asset Support** - Supports BTC, ETH, SOL, and other assets

```python
from utils.local_data_loader import LocalDataLoader

# Initialize loader
loader = LocalDataLoader()

# Get available assets
assets = loader.get_available_assets()  # ['BTC', 'ETH', 'SOL']

# Get available factors
factors = loader.get_available_factors('BTC')

# Load data pair
price_data, factor_data = loader.load_data_pair('BTC', 'factor_name')
```

### Automated Data Download
```bash
# Start automated data download
python scripts/daily_download.py
```

### Data Verification
```bash
# Check API connectivity and current prices
python tools/simple_price_check.py
```

### Generate Metadata Catalog
```bash
# Generate metadata catalog (one-time setup)
python tools/generate_metadata_catalog.py
```

## Backtesting

### Performance Metrics
The backtesting engine calculates comprehensive performance metrics:

#### Core Metrics
- **Total Return** - Overall percentage return over the backtest period
- **Annual Return** - Annualized return rate (daily return × 365)
- **Sharpe Ratio** - Risk-adjusted return measure (return / volatility × √365)
- **Maximum Drawdown** - Largest peak-to-trough decline percentage
- **Maximum Drawdown (Dollar)** - Largest peak-to-trough decline in dollar terms

#### Trading Metrics
- **Number of Trades** - Total number of completed trades
- **Win Rate** - Percentage of profitable trades
- **Average Position Duration** - Average time positions are held
- **Calmar Ratio** - Annual return / maximum drawdown

#### Buy & Hold Comparison
- **Buy & Hold Return** - Performance of buy-and-hold strategy
- **Strategy vs Buy & Hold** - Relative performance comparison
- **Excess Return** - Strategy return minus buy & hold return

### Interactive Trading System
```bash
# Start interactive trading session (recommended)
python scripts/interactive_trading.py
```

**Features:**
- **Interactive Interface** - User-friendly parameter input
- **Multiple Operations** - Backtest, optimization, or both
- **Data Source Selection** - Choose between local CSV files or API data
- **Parameter Validation** - Built-in validation for all inputs
- **Factor Search** - Search and select from available factors

### Automated LLM Backtest Agent
```bash
# Run automated daily backtest agent
python scripts/llm_backtest_agent.py

# Schedule daily execution (Windows)
python scripts/schedule_daily_agent.py
```

**Features:**
- **Automated Processing** - Processes all factors automatically
- **LLM-Powered Analysis** - Intelligent factor screening and optimization
- **Daily Reports** - Generates comprehensive daily reports
- **Performance Monitoring** - Tracks system performance

### Quick Testing Tools
```bash
# Quick parameter testing (for developers)
python tools/simple_optimization.py

# Legacy algorithm verification
python tools/legacy_algorithm.py
```

**Tool Purposes:**
- **`tools/simple_optimization.py`** - Quick parameter changes without interactive interface
- **`tools/legacy_algorithm.py`** - Legacy algorithm file to ensure logic and results match the old system
- **`tools/simple_price_check.py`** - Verify API connectivity and check current prices
- **`tools/generate_metadata_catalog.py`** - Generate metadata catalog from Glassnode API

## Live Trading

### Interactive Trading System
```bash
# Start interactive trading session
python scripts/interactive_trading.py
```

**Trading Features:**
- **Real-time Data** - Live market data integration
- **Risk Management** - Built-in position sizing and risk controls
- **Multiple Assets** - Support for BTC, ETH, SOL, TON, TRX, USDC, USDT
- **Strategy Integration** - Seamless integration with backtesting strategies

## Complete Example Workflow

### Step 1: Data Download and Setup
```bash
# 1. Set up environment and API keys
python scripts/setup_environment.py

# 2. Verify API connectivity
python tools/simple_price_check.py

# 3. Generate metadata catalog (one-time)
python tools/generate_metadata_catalog.py

# 4. Download historical data
python scripts/daily_download.py
```

### Step 2: Strategy Development and Testing
```bash
# 1. Quick parameter testing
python tools/simple_optimization.py

# 2. Interactive backtesting and optimization
python scripts/interactive_trading.py

# 3. Verify results against legacy algorithm
python tools/legacy_algorithm.py
```

### Step 3: Automated Daily Backtesting (Optional)
```bash
# 1. Configure LLM agent settings
# Edit config/llm_agent_config.json

# 2. Run daily agent
python scripts/llm_backtest_agent.py

# 3. Schedule daily execution (Windows)
python scripts/schedule_daily_agent.py
```

### Step 4: Live Trading (Optional)
```bash
# 1. Configure Bybit API credentials in .env
# 2. Test with small position sizes
# 3. Monitor performance and adjust parameters
```

## Key Components

### Data Loader (`utils/local_data_loader.py`)
- Intelligent CSV file processing
- Automatic data format detection and standardization
- Comprehensive data validation
- Date overlap verification

### Backtesting Engine (`core/enhanced_engine.py`)
- High-performance backtesting execution
- Multiple performance metrics calculation
- Support for custom strategies
- Dynamic position sizing

### Enhanced Non-Price Strategy (`strategies/enhanced_non_price_strategy.py`)
- Advanced strategy using on-chain metrics
- Configurable parameters
- Real-time signal generation
- Parameter optimization capabilities

### LLM Components
- **LLM Client** (`core/llm_client.py`) - LLM API integration
- **LLM Scheduler** (`core/llm_scheduler.py`) - Intelligent task scheduling
- **Factor Screening** (`core/factor_screening.py`) - Two-stage factor screening
- **Context Manager** (`core/context_manager.py`) - Context management
- **Parameter Generator** (`core/intelligent_param_generator.py`) - LLM-powered parameter generation

### Trading Interface (`trading/`)
- Bybit exchange integration
- Order management and execution
- Risk management features

## Configuration

### Environment Variables
- `GLASSNODE_API_KEY` - Primary Glassnode API key
- `GLASSNODE_API_KEY_BACKUP` - Backup API key
- `BYBIT_API_KEY` - Bybit API key for live trading
- `BYBIT_SECRET` - Bybit API secret

### Configuration Files
- `config/download_config.py` - Download rate limiting configuration
- `config/trading_config.py` - Trading configuration
- `config/paths.py` - Path configuration
- `config/llm_agent_config.json` - LLM agent configuration (if using LLM features)

### Data Paths
- Default data storage: `D:\Trading_Data\glassnode_data2`
- Configurable via `config/paths.py`

## Performance Monitoring

### Logging
- Comprehensive logging to both console and files
- Session-based progress tracking
- Error reporting and debugging
- Configurable verbosity levels

### Reports
- Optimization results stored in `optimization_reports/`
- Structured organization by asset and factor
- Performance metrics and sensitivity analysis
- Strategy comparison reports
- Daily agent reports (if using LLM agent)

## Important Notes

### API Usage
- Respect Glassnode API rate limits
- Use multiple API keys for high-volume operations
- Monitor API usage and costs
- Configure rate limiting in `config/download_config.py`

### Risk Management
- Always test strategies thoroughly before live trading
- Start with small position sizes
- Monitor positions and set stop-losses

### Data Quality
- Verify data integrity before trading
- Check for missing or corrupted data
- Regular data validation and cleaning
- Ensure data path is correctly set to `D:\Trading_Data\glassnode_data2`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## Updates

- **v2.2** - Code optimization and reorganization (Phase 1 & 2)
- **v2.1** - Enhanced optimization system with LLM integration
- **v2.0** - Enhanced data downloader with session management
- **v1.5** - Added real-time trading capabilities
- **v1.0** - Initial release with backtesting engine

## Next Steps

### Development Roadmap

#### Strategy Enhancement
- **Multi-Factor Strategy Integration** - Combine multiple on-chain metrics for improved signal generation
- **Strategy Manager Integration** - Integrate strategy manager with the main system for better strategy management

#### Performance Analysis
- **Performance Graph Reporting** - Add comprehensive visualization tools:
  - Heatmap analysis for parameter sensitivity
  - Long-short trade count distribution
  - Strategy PnL table with detailed breakdown
  - Equity curve visualization
  - Drawdown analysis charts

#### Infrastructure & Deployment
- **AWS Lightsail Deployment** - Deploy the system on AWS Lightsail for cloud-based operation
- **Real-Time Trading Integration** - Enhance live trading capabilities with:
  - Real-time market data streaming
  - Automated order execution
  - Risk management automation
  - Performance monitoring dashboard

#### System Integration
- **Strategy Manager System** - Complete integration of strategy manager with the main trading system
- **Real-Time Trading Platform** - Full deployment of real-time trading capabilities

---

**Disclaimer**: This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk. Always perform your own research and consider consulting with financial advisors before making trading decisions.
