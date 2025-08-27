# 🚀 Advanced Nifty Options Trading Bot v2.0

A sophisticated, AI-powered trading bot for automated Nifty and Bank Nifty options trading with adaptive learning across 16 market conditions.

## 🌟 Features

### 🧠 **Adaptive Learning System**
- **16 Market Conditions**: Comprehensive analysis covering all market scenarios
- **Dual Strategy Engine**: Adaptive strategy + Neural Network ML models
- **Real-time Learning**: Continuous improvement based on performance feedback
- **Dynamic Weight Adjustment**: Strategies adapt based on success rates

### 📊 **Advanced Technical Analysis**
- **Multi-timeframe Analysis**: Daily and 1-minute data integration
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Pattern Recognition**: Candlestick patterns and trend analysis
- **Gap Detection**: Automated gap analysis and fade strategies
- **Support/Resistance**: Dynamic level identification

### ⚡ **Professional Trading Engine**
- **Risk Management**: Multi-layer risk controls with adaptive position sizing
- **Real-time Monitoring**: Live position tracking and P&L updates
- **Options Chain Analysis**: Intelligent strike selection based on liquidity and premium
- **Multiple Strategies**: Trend following, gap fade, volatility, and range-bound strategies

### 🖥️ **Desktop Application**
- **Modern GUI**: Professional desktop interface with real-time updates
- **Live Dashboard**: Market conditions, positions, and performance metrics
- **Configuration Panel**: Easy setup and API management
- **Backtesting Engine**: Historical strategy validation
- **Performance Analytics**: Detailed performance tracking by market condition

## 📁 Project Structure

```
trading-bot/
├── 🎮 app.py                    # Desktop GUI launcher
├── 📋 requirements.txt          # Dependencies
├── 📖 README.md                # This file
├── 🔧 .env.example             # Environment variables template
│
├── 🏗️ core/                    # Core trading logic
│   ├── __init__.py
│   ├── trading_engine.py       # Main trading engine with 16 conditions
│   ├── risk_manager.py         # Advanced risk management
│   └── position_manager.py     # Position handling and execution
│
├── 🧠 strategies/              # Trading strategies
│   ├── __init__.py
│   ├── adaptive_strategy.py    # Adaptive learning strategy
│   └── ml_strategy.py          # Neural network ML strategy
│
├── 📊 data/                    # Data management
│   ├── __init__.py
│   ├── data_manager.py         # CSV backups and data handling
│   └── market_analyzer.py      # Technical analysis engine
│
├── 🔌 api/                     # API integrations
│   ├── __init__.py
│   └── fyers_client.py         # Fyers API client
│
├── ⚙️ config/                  # Configuration
│   ├── __init__.py
│   ├── config_manager.py       # Centralized config management
│   └── config.json             # Configuration file
│
├── 🛠️ utils/                   # Utilities
│   ├── __init__.py
│   └── logger.py               # Enhanced logging system
│
├── 📈 backtest_engine.py       # Backtesting framework
└── 📝 logs/                    # Log files
    ├── trading_engine.log
    ├── data_manager.log
    └── performance.log
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone or download the project
cd trading-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env` with your Fyers API credentials:
```env
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_ACCESS_TOKEN=your_access_token
FYERS_REDIRECT_URI=https://trade.fyers.in/api-login/redirect-to-app
```

### 3. **Launch Desktop App**

```bash
python app.py
```

### 4. **Command Line Usage**

```bash
# For command line trading (optional)
python -m core.trading_engine
```

## 🎯 16 Market Conditions

The bot intelligently identifies and adapts to these market conditions:

| Condition | Strategy | Learning Focus |
|-----------|----------|----------------|
| 🟢 **Strong Bull Trend** | Sell OTM Puts | Trend strength validation |
| 🟢 **Weak Bull Trend** | Conservative Put selling | Risk-adjusted positioning |
| 🔴 **Strong Bear Trend** | Sell OTM Calls | Downtrend confirmation |
| 🔴 **Weak Bear Trend** | Conservative Call selling | Reversal probability |
| ↔️ **Sideways Range** | Iron Condor/Strangles | Range boundaries |
| ⬆️ **Breakout Up** | Momentum Put selling | Breakout sustainability |
| ⬇️ **Breakout Down** | Momentum Call selling | Volume confirmation |
| 📈 **Gap Up** | Gap fade Call selling | Gap fill probability |
| 📉 **Gap Down** | Gap fade Put selling | Support levels |
| 🌪️ **High Volatility** | Premium selling strategies | IV crush timing |
| 😴 **Low Volatility** | Volatility expansion plays | Catalyst identification |
| 🔄 **Reversal Bullish** | Early Put selling | Reversal confirmation |
| 🔄 **Reversal Bearish** | Early Call selling | Pattern completion |
| 📦 **Consolidation** | Range-bound strategies | Breakout preparation |
| 📰 **News Driven** | Event-based positioning | News impact assessment |
| ⏰ **Expiry Day** | Time decay strategies | Greeks optimization |

## 🧠 Adaptive Learning System

### **Strategy Learning**
- **Performance Tracking**: Success rate by market condition
- **Weight Adjustment**: Dynamic strategy weighting based on results
- **Confidence Scoring**: ML-based confidence in signal generation
- **Risk Adaptation**: Position sizing based on historical performance

### **ML Model Features**
- **Random Forest**: Pattern recognition and trend analysis
- **Neural Network**: Deep learning for complex market relationships
- **Feature Engineering**: 17+ technical and fundamental features
- **Continuous Training**: Models retrain with new market data

## 📊 Risk Management

### **Multi-Layer Protection**
- **Position Limits**: Maximum positions and exposure controls
- **Daily Loss Limits**: Automatic trading halt on loss thresholds
- **Dynamic Stop Loss**: Adaptive stop loss based on market conditions
- **Correlation Limits**: Prevent over-concentration in similar trades

### **Adaptive Position Sizing**
- **Volatility Adjusted**: Position size scales with market volatility
- **Confidence Based**: Larger positions for higher confidence signals
- **Performance Weighted**: Size adjusts based on strategy success rate

## 🖥️ Desktop Application Features

### **Control Panel**
- ▶️ Start/Stop bot with one click
- 📊 Real-time market condition display
- 💼 Live position monitoring
- 📈 P&L tracking

### **Configuration**
- 🔧 API credentials management
- ⚙️ Risk parameter adjustment
- 🎯 Trading symbol selection
- ⏱️ Cycle interval configuration

### **Performance Analytics**
- 📈 Total and daily P&L
- 🎯 Win rate by market condition
- 📊 Strategy performance comparison
- 📋 Detailed trade history

### **Backtesting**
- 📊 Historical strategy validation
- 📈 Performance metrics calculation
- 📋 Detailed backtest reports
- 💾 Results export

## 📈 Backtesting

Run comprehensive backtests:

```python
from backtest_engine import BacktestEngine

# Initialize backtest
backtest = BacktestEngine(config)

# Run 3-month backtest
results = backtest.run_backtest(
    start_date='2024-05-01',
    end_date='2024-08-01'
)

# View results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## 📊 Data Management

### **Automated Backups**
- 💾 Historical data saved to CSV
- 📋 Trade logs with complete details
- 📊 Performance metrics tracking
- 🗂️ Organized data structure

### **Data Structure**
```
data/
├── historical/          # OHLCV data
├── options/            # Options chain data
├── trades/             # Trade execution logs
├── analysis/           # Technical analysis results
├── backtests/          # Backtest results
└── performance/        # Performance reports
```

## ⚠️ Important Notes

### **Risk Disclaimer**
- Trading involves substantial risk of loss
- Past performance doesn't guarantee future results
- Use paper trading first to validate strategies
- Never risk more than you can afford to lose

### **API Requirements**
- Valid Fyers trading account required
- API access must be enabled
- Sufficient margin for options trading
- Real-time data subscription recommended

### **System Requirements**
- Python 3.8 or higher
- Windows/Linux/macOS
- Stable internet connection
- Minimum 4GB RAM recommended

## 🔧 Advanced Configuration

### **Environment Variables**
```env
# Fyers API
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_ACCESS_TOKEN=your_access_token

# Risk Management
MAX_DAILY_LOSS=10000
MAX_POSITIONS=3
PORTFOLIO_VALUE=100000

# Trading
SYMBOLS=NIFTY,BANKNIFTY
CYCLE_INTERVAL=5
```

### **Config File Options**
```json
{
  "risk_management": {
    "max_daily_loss": 10000,
    "max_positions": 3,
    "stop_loss_pct": 200,
    "profit_target_pct": 50
  },
  "ml_model": {
    "retrain_interval": 24,
    "min_data_points": 100
  },
  "trading": {
    "symbols": ["NIFTY", "BANKNIFTY"],
    "cycle_interval": 5
  }
}
```

## 🤝 Support

For issues, questions, or contributions:
- 📧 Create an issue in the repository
- 📖 Check the documentation
- 💬 Join the community discussions

## 📄 License

This project is for educational and research purposes. Please ensure compliance with local trading regulations.

---

**⚡ Ready to revolutionize your options trading with AI? Launch the desktop app and start your journey to automated trading success!**
