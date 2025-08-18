# Algorithmic Trading System

A comprehensive algorithmic trading platform for cryptocurrency markets, featuring real-time data collection, backtesting, strategy optimization, and live trading capabilities.

## 🚀 Features

### Core Functionality
- **Real-time Data Collection** - Automated Glassnode API data downloading with rate limiting and error handling
- **Advanced Backtesting Engine** - High-performance backtesting with multiple performance metrics
- **Strategy Optimization** - Parameter optimization with sensitivity analysis and structured output
- **Live Trading** - Real-time trading execution on Bybit exchange
- **Data Management** - Comprehensive data validation, cleaning, and storage with intelligent CSV processing

### Data Sources
- **Glassnode API** - On-chain and market data for cryptocurrencies
- **Multiple API Keys** - Automatic key rotation and fallback mechanisms
- **Local Data Storage** - Efficient CSV-based data storage with incremental updates

### Trading Strategies
- **Non-Price Strategy** - Advanced strategy using on-chain metrics and market indicators
- **Modular Architecture** - Easy to extend with new strategies
- **Parameter Optimization** - Automated strategy parameter tuning with user confirmation

## 📁 Project Structure

```
Algo_Trading/
├── config/                        # Configuration and settings
│   ├── __init__.py               # Config module entry
│   ├── paths.py                  # Path configuration
│   ├── settings.py               # System settings
│   └── secrets.py                # Encrypted API key management
├── core/                         # Core engine components
│   ├── __init__.py               # Core module entry
│   ├── dataloader.py             # Glassnode API data loader
│   └── engine.py                 # Backtesting engine
├── scripts/                      # Utility and execution scripts
│   ├── daily_download.py         # Data downloader
│   ├── interactive_trading.py    # Interactive trading interface
│   ├── generate_metadata_catalog.py # Metadata catalog generation
│   ├── download_config.py        # Download configuration
│   └── setup_environment.py      # Environment setup
├── tools/                        # Development and testing tools
│   ├── legacy_algorithm.py       # Legacy algorithm file (for logic verification)
│   ├── simple_price_check.py     # API connectivity test
│   └── simple_optimization.py    # Quick parameter testing
├── strategies/                   # Trading strategies
│   ├── __init__.py               # Strategies module entry
│   └── non_price_strategy.py     # Non-price based strategy
├── trading/                      # Live trading components
│   ├── brokers/                  # Exchange connectivity
│   ├── execution/                # Order execution
│   └── real_time/                # Real-time trading
├── utils/                        # Utility functions
│   ├── local_data_loader.py      # Intelligent local data management
│   └── strategy_manager.py       # Strategy management
├── optimization_reports/         # Optimization results (structured)
└── notebooks/                    # Jupyter notebooks
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Glassnode API key(s)
- Bybit API credentials (for live trading)

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

## 📊 Data Management

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

## 🔬 Backtesting

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

## 💹 Live Trading

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

## 📋 Complete Example Workflow

### Step 1: Data Download and Setup
```bash
# 1. Set up environment and API keys
python scripts/setup_environment.py

# 2. Verify API connectivity
python tools/simple_price_check.py

# 3. Download historical data
python scripts/daily_download.py

# 4. Generate metadata catalog (optional)
python scripts/generate_metadata_catalog.py
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

### Step 3: Live Trading (Optional)
```bash
# 1. Configure Bybit API credentials in .env
# 2. Test with small position sizes
# 3. Monitor performance and adjust parameters
```

### Example Interactive Session
```bash
$ python scripts/interactive_trading.py

=== ALGORITHMIC TRADING SYSTEM ===
1. Backtest only
2. Optimization only  
3. Both backtest and optimization

Select operation (1-3): 1

=== ASSET SELECTION ===
Available assets: ['BTC', 'ETH', 'SOL']
Select asset: BTC

=== FACTOR SELECTION ===
Available factors for BTC: ['mvrv', 'nvt', 'sopr', ...]
Select factor: mvrv

=== DATA SOURCE SELECTION ===
1. Local CSV files
2. API data
Select data source (1-2): 1

=== PARAMETER CONFIGURATION ===
Long parameter (default 0.02): 0.02
Short parameter (default -0.02): -0.02
Rolling window (default 1): 1

=== BACKTEST RESULTS ===
Total Return: 15.23%
Annual Return: 12.45%
Sharpe Ratio: 1.85
Maximum Drawdown: 8.67%
Number of Trades: 45
```

## 🛠️ Development Tools

### Tools Directory (`tools/`)
The `tools/` directory contains specialized scripts for development, testing, and verification:

- **`legacy_algorithm.py`** - Legacy algorithm implementation
  - Contains the original trading logic for verification
  - Ensures new implementations produce identical results
  - Used for regression testing and logic validation

- **`simple_price_check.py`** - API connectivity verification
  - Tests Glassnode API connectivity
  - Displays current cryptocurrency prices
  - Validates API key configuration

- **`simple_optimization.py`** - Quick parameter testing
  - Allows rapid parameter changes without interactive interface
  - Ideal for developers and automated testing
  - Hardcoded parameters for quick iteration

## 📈 Key Components

### Data Loader (`utils/local_data_loader.py`)
- Intelligent CSV file processing
- Automatic data format detection and standardization
- Comprehensive data validation
- Date overlap verification

### Backtesting Engine (`core/engine.py`)
- High-performance backtesting execution
- Multiple performance metrics calculation
- Support for custom strategies

### Non-Price Strategy (`strategies/non_price_strategy.py`)
- Advanced strategy using on-chain metrics
- Configurable parameters
- Real-time signal generation
- Parameter optimization capabilities

### Trading Interface (`trading/`)
- Bybit exchange integration
- Order management and execution
- Risk management features

## 🔧 Configuration

### Environment Variables
- `GLASSNODE_API_KEY` - Primary Glassnode API key
- `GLASSNODE_API_KEY_BACKUP` - Backup API key
- `BYBIT_API_KEY` - Bybit API key for live trading
- `BYBIT_SECRET` - Bybit API secret

### Data Paths
- Default data storage: `D:\Trading_Data\glassnode_data2`
- Configurable via `config/paths.py`

## 📊 Performance Monitoring

### Logging
- Comprehensive logging to both console and files
- Session-based progress tracking
- Error reporting and debugging

### Reports
- Optimization results stored in `optimization_reports/`
- Structured organization by asset and factor
- Performance metrics and sensitivity analysis
- Strategy comparison reports

## 🚨 Important Notes

### API Usage
- Respect Glassnode API rate limits
- Use multiple API keys for high-volume operations
- Monitor API usage and costs

### Risk Management
- Always test strategies thoroughly before live trading
- Start with small position sizes
- Monitor positions and set stop-losses

### Data Quality
- Verify data integrity before trading
- Check for missing or corrupted data
- Regular data validation and cleaning
- Ensure data path is correctly set to `D:\Trading_Data\glassnode_data2`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Algorithmic Trading System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🔄 Updates

- **v2.1** - Enhanced optimization system
- **v2.0** - Enhanced data downloader with session management
- **v1.5** - Added real-time trading capabilities
- **v1.0** - Initial release with backtesting engine

## 🚀 Next Steps

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