# Jupyter Notebooks

This directory contains Jupyter notebooks for interactive data analysis and visualization in the algorithmic trading system.

## 📊 Available Notebooks

### TradingDataInteractiveViewer.ipynb
An advanced interactive data viewer for exploring trading data with comprehensive visualization capabilities.

**Key Features:**
- **Interactive Data Visualization** - Real-time plotting with Plotly
- **Multi-Asset Support** - View data for BTC, ETH, SOL, and other assets
- **Factor Analysis** - Explore on-chain metrics and indicators
- **Statistical Analysis** - Always-on statistics panel with factor/pct/abs change metrics
- **Advanced Controls** - Multiple aggregation methods, normalization, and filtering options

**Main Components:**
- **Asset Selection** - Dropdown to choose cryptocurrency asset
- **Factor Multi-Selection** - Search and select multiple factors for comparison
- **Date Range Control** - Interactive date slider and manual date picker
- **Visualization Options** - Frequency, scale, normalization, and aggregation controls
- **Technical Indicators** - SMA, EMA, Median with customizable periods
- **Change Analysis** - Percentage and absolute change calculations
- **Statistics Panel** - Real-time statistical summaries for selected data
- **Interactive Plotting** - Click to add horizontal lines, zoom, pan, and hover

## 🚀 Getting Started

### Prerequisites
```bash
# Install Jupyter and required packages
pip install jupyter pandas numpy matplotlib seaborn plotly

# Start Jupyter server
jupyter notebook
```

### Environment Setup
1. **Activate your virtual environment** (if using one)
2. **Set up API keys** in your environment variables
3. **Configure data paths** in `config/paths.py`
4. **Start Jupyter** from the project root directory

### Basic Usage
```python
# Import project modules
import sys
sys.path.append('..')

from utils.local_data_loader import LocalDataLoader
from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
from core.enhanced_engine import EnhancedBacktestEngine

# Load data
loader = LocalDataLoader()
price_data, factor_data = loader.load_data_pair('BTC', 'mvrv')

# Run analysis
# ... your notebook code here
```

## 📈 Using TradingDataInteractiveViewer

### Setup Requirements
```bash
# Install required packages
pip install jupyter pandas numpy plotly ipywidgets scipy

# Ensure data path is correct
# The notebook expects data at: D:\Trading_Data\glassnode_data2
```

### Data Structure
The viewer expects the following data structure:
```
D:\Trading_Data\glassnode_data2\
├── BTC\
│   ├── *_market_price_usd_close_*.csv    # Price data
│   ├── factor1_*.csv                     # Factor data
│   ├── factor2_*.csv                     # Factor data
│   └── ...
├── ETH\
│   ├── *_market_price_usd_close_*.csv
│   └── ...
└── ...
```

### Key Features Explained

#### 1. Asset and Factor Selection
- **Asset Dropdown**: Choose from available cryptocurrencies (BTC, ETH, SOL, etc.)
- **Factor Search**: Type to filter available factors
- **Multi-Selection**: Select multiple factors for comparison

#### 2. Data Processing Options
- **Frequency**: raw, 5min, 15min, 30min, 1H, 2H, 4H, 12H, 1D, 1W
- **Scale**: linear or logarithmic
- **Aggregation**: mean, last, or sum for factor data
- **Normalization**: none, zscore, or minmax

#### 3. Technical Indicators
- **SMA (Simple Moving Average)**: Customizable period (2-400)
- **EMA (Exponential Moving Average)**: Customizable period (2-400)
- **MM (Median)**: Customizable period (2-600)

#### 4. Change Analysis
- **Percentage Change**: Calculate percentage changes over specified periods
- **Absolute Change**: Calculate absolute value changes over specified periods

#### 5. Statistics Panel
Toggle between three statistical views:
- **Factor Statistics**: Basic statistics of raw factor values
- **Pct Change Statistics**: Statistics of percentage changes
- **Abs Change Statistics**: Statistics of absolute changes

#### 6. Interactive Features
- **Date Range Slider**: Interactive date selection
- **Manual Date Picker**: Precise date range selection
- **Horizontal Lines**: Click on plot to add reference lines
- **Zoom and Pan**: Interactive plot navigation

## 🔧 Technical Features

### Data Processing
- **Smart CSV Parsing** - Automatically detects time/value columns
- **Epoch Time Support** - Handles various timestamp formats (seconds, milliseconds, microseconds)
- **Data Validation** - Checks for missing values and data integrity
- **Resampling** - Flexible time-based data aggregation

### Visualization Engine
- **Plotly Integration** - Interactive charts with zoom, pan, and hover
- **Dual Y-Axis** - Price on primary axis, factors on secondary axis
- **Real-time Updates** - Dynamic plot updates based on user selections
- **Custom Styling** - Configurable colors, line styles, and layouts

### Statistical Analysis
- **Comprehensive Statistics** - Mean, std, min, max, quartiles, count, missing values
- **Multiple Views** - Factor, percentage change, and absolute change statistics
- **Real-time Calculation** - Statistics update automatically with data changes

## 📊 Statistical Metrics

The notebook provides comprehensive statistical analysis for the selected data:

### Basic Statistics
- **Count** - Number of data points
- **Missing** - Number of missing values
- **Mean** - Average value
- **Std** - Standard deviation
- **Min** - Minimum value
- **25%** - First quartile
- **Median** - Middle value
- **75%** - Third quartile
- **Max** - Maximum value

### Change Analysis
- **Percentage Change** - Relative change over specified periods
- **Absolute Change** - Absolute value change over specified periods
- **Period Customization** - Configurable periods (1-50) for change calculations

## 🛠️ Development Guidelines

### Code Organization
- **Clear Structure** - Use markdown cells for documentation
- **Modular Code** - Break complex analysis into functions
- **Reusable Components** - Create utility functions for common tasks
- **Version Control** - Track changes with Git

### Best Practices
- **Documentation** - Explain analysis steps and assumptions
- **Error Handling** - Handle data loading and API errors gracefully
- **Performance** - Optimize code for large datasets
- **Reproducibility** - Set random seeds and document dependencies

### Data Management
- **Data Validation** - Check data quality before analysis
- **Caching** - Cache expensive computations
- **Memory Management** - Handle large datasets efficiently
- **Backup** - Save intermediate results and plots

## 📁 File Structure

```
notebooks/
├── README.md                           # This file
└── TradingDataInteractiveViewer.ipynb  # Interactive data viewer
```

## 🔍 Usage Workflow

### 1. Launch the Notebook
```bash
# Start Jupyter from project root
jupyter notebook notebooks/TradingDataInteractiveViewer.ipynb
```

### 2. Configure Data Path
The notebook expects data at `D:\Trading_Data\glassnode_data2`. If your data is elsewhere, modify the `ROOT` variable:
```python
ROOT = Path(r"your_data_path_here")
```

### 3. Interactive Usage
1. **Select Asset**: Choose from available cryptocurrencies
2. **Search Factors**: Type to filter available factors
3. **Select Factors**: Multi-select factors for comparison
4. **Adjust Settings**: Configure frequency, scale, normalization
5. **Set Date Range**: Use slider or manual date picker
6. **Add Indicators**: Enable SMA, EMA, or Median with custom periods
7. **View Statistics**: Toggle between factor/pct/abs change statistics
8. **Interact with Plot**: Zoom, pan, add horizontal lines by clicking

### 4. Data Export
The notebook provides interactive visualization. For data export, use the project's main scripts:
```bash
python scripts/interactive_trading.py  # For backtesting
python tools/simple_optimization.py    # For quick analysis
```

## 🚨 Important Notes

### Data Requirements
- **Data Path**: Ensure data is available at `D:\Trading_Data\glassnode_data2`
- **File Format**: CSV files with time/value columns (t/v or timestamp/value)
- **Price Files**: Must contain `*_market_price_usd_close_*.csv` pattern
- **Factor Files**: Any other CSV files in asset folders

### Performance Considerations
- **Large Datasets**: The viewer handles large datasets efficiently with smart sampling
- **Memory Usage**: Monitor memory usage when loading many factors simultaneously
- **Rendering**: Plot updates may take time with complex configurations

### Browser Compatibility
- **Jupyter Lab**: Recommended for best interactive experience
- **Jupyter Notebook**: Compatible but may have limited widget support
- **Plotly Renderer**: Uses `plotly_mimetype` for inline display

## 🤝 Contributing

### Adding New Notebooks
1. **Follow Naming Convention** - Use descriptive names
2. **Include Documentation** - Add markdown cells explaining functionality
3. **Test Thoroughly** - Ensure notebooks run without errors
4. **Update This README** - Add new notebooks to the documentation

### Development Guidelines
- **Widget Integration** - Use ipywidgets for interactive components
- **Error Handling** - Include proper error handling for data loading
- **Performance** - Optimize for large datasets
- **Documentation** - Document all interactive features and controls

## 📚 Additional Resources

### Documentation
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Plotly Documentation](https://plotly.com/python/)
- [IPyWidgets Documentation](https://ipywidgets.readthedocs.io/)

### Libraries Used
- **Plotly**: Interactive plotting and visualization
- **IPyWidgets**: Interactive widgets and controls
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing utilities

### Community
- [Jupyter Community](https://jupyter.org/community)
- [Plotly Community](https://community.plotly.com/)

---

**Note**: These notebooks are for educational and research purposes. Always validate results and consider consulting with financial advisors before making trading decisions.
