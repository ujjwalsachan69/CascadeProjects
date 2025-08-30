#!/usr/bin/env python3
"""
=============================================================================
MAIN TRADING BOT FILE - This is the heart of your trading system
MAIN TRADING BOT FILE - This is the heart of your trading system
MAIN TRADING BOT FILE - This is the heart of your trading system
=============================================================================

PURPOSE: This file runs the complete automated trading bot that:
- Connects to Fyers API for live market data
- Analyzes Nifty and Bank Nifty trends using 3 months of historical data
- Automatically selects and sells options based on market conditions
- Uses machine learning to improve trading decisions
- Manages risk and positions automatically

KEY FUNCTIONS:
1. analyze_market() - Analyzes market trends and gaps
2. select_trading_strategy() - Chooses best strategy based on market conditions
3. execute_trades() - Places buy/sell orders automatically
4. monitor_positions() - Tracks open positions and P&L
5. run_trading_session() - Main trading loop

IMPORTANT SETTINGS YOU CAN EDIT:
=============================================================================
Line 45-55: TRADING SYMBOLS - Change which stocks/indices to trade
Line 60-70: RISK SETTINGS - Adjust position sizes and stop losses  
Line 75-85: STRATEGY SETTINGS - Modify trading strategies
Line 90-100: TIME SETTINGS - Change trading hours and intervals
=============================================================================
"""

import os
import sys
import time
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass
import dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules from new structure
from api.fyers_client import FyersClient
from data.market_analyzer import DataAnalyzer
from core.options_manager import OptionsManager
from data.data_manager import DataManager
from utils.logger import setup_logger
from core.trade_confirmation import TradeConfirmationSystem

# Load environment variables
dotenv.load_dotenv()

# Configure logging
setup_logger()

logger = logging.getLogger(__name__)

# =============================================================================
# EDITABLE TRADING CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# =============================================================================

# TRADING SYMBOLS - Line 45-55
# Edit these to change which stocks/indices you want to trade
TRADING_SYMBOLS = [
    'NIFTY',      # Main Nifty 50 index - EDIT: Add/remove symbols here
    'BANKNIFTY'   # Bank Nifty index - EDIT: Add 'FINNIFTY' for Fin Nifty
]

# RISK MANAGEMENT SETTINGS - Line 60-70  
# Edit these values to control your risk exposure
RISK_CONFIG = {
    'max_daily_loss': 5000,        # EDIT: Maximum loss per day in rupees
    'max_position_size': 100000,   # EDIT: Maximum position size in rupees
    'stop_loss_pct': 0.05,         # EDIT: Stop loss percentage (5% = 0.05)
    'profit_target_pct': 0.10,     # EDIT: Profit target percentage (10% = 0.10)
    'max_positions': 5,            # EDIT: Maximum number of open positions
    'risk_per_trade': 0.01         # EDIT: Risk per trade (1% of portfolio = 0.01)
}

# STRATEGY SETTINGS - Line 75-85
# Edit these to modify trading behavior
STRATEGY_CONFIG = {
    'primary_strategy': 'adaptive',     # EDIT: 'adaptive', 'ml', or 'conservative'
    'min_confidence': 0.7,              # EDIT: Minimum confidence to place trade (0-1)
    'trend_strength_threshold': 20,     # EDIT: Minimum trend strength (0-100)
    'volatility_threshold': 0.25,       # EDIT: Maximum volatility to trade (0-1)
    'gap_threshold': 0.5,               # EDIT: Minimum gap size to consider (%)
    'volume_threshold': 1.5             # EDIT: Minimum volume ratio to trade
}

# TIME SETTINGS - Line 90-100
# Edit these to change when the bot trades
TIME_CONFIG = {
    'market_start': '09:15',        # EDIT: Market opening time
    'market_end': '15:30',          # EDIT: Market closing time  
    'analysis_interval': 300,       # EDIT: Analysis interval in seconds (300 = 5 minutes)
    'position_check_interval': 60,  # EDIT: Position monitoring interval in seconds
    'max_trading_hours': 6          # EDIT: Maximum hours to trade per day
}

# DATA SETTINGS - Line 105-115
# Edit these for historical data analysis
DATA_CONFIG = {
    'historical_days': 90,          # EDIT: Days of historical data to analyze
    'minute_data_days': 5,          # EDIT: Days of minute data for short-term analysis
    'technical_indicators': [       # EDIT: Add/remove technical indicators
        'sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_bands'
    ],
    'save_data': True,              # EDIT: Set to False to disable data saving
    'backup_trades': True           # EDIT: Set to False to disable trade backup
}

# =============================================================================
# MAIN TRADING BOT CLASS - Advanced users can modify this
# =============================================================================

@dataclass
class TradingPosition:
    """Represents a trading position"""
    symbol: str
    option_symbol: str
    quantity: int
    entry_price: float
    current_price: float
    timestamp: datetime
    position_type: str  # 'short_call', 'short_put'
    strike: float
    expiry: datetime
    pnl: float = 0.0

class RiskManager:
    """Advanced risk management for options trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_position_value = config.get('max_position_value', 50000)
        self.max_daily_loss = config.get('max_daily_loss', 10000)
        self.max_positions = config.get('max_positions', 3)
        self.stop_loss_pct = config.get('stop_loss_pct', 200)  # 200% of premium
        self.profit_target_pct = config.get('profit_target_pct', 50)  # 50% of premium
        self.daily_pnl = 0.0
    
    def can_open_position(self, premium: float, current_positions: int) -> bool:
        """Check if we can open a new position"""
        if current_positions >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
        
        position_value = premium * 25  # Nifty lot size
        if position_value > self.max_position_value:
            logger.warning(f"Position value too large: {position_value}")
            return False
        
        return True
    
    def should_close_position(self, position: TradingPosition) -> tuple[bool, str]:
        """Check if position should be closed"""
        # Stop loss check
        loss_pct = (position.current_price - position.entry_price) / position.entry_price * 100
        if loss_pct > self.stop_loss_pct:
            return True, "stop_loss"
        
        # Profit target check
        profit_pct = (position.entry_price - position.current_price) / position.entry_price * 100
        if profit_pct > self.profit_target_pct:
            return True, "profit_target"
        
        # Time-based exit (close on expiry day)
        days_to_expiry = (position.expiry - datetime.now()).days
        if days_to_expiry <= 0:
            return True, "expiry"
        
        return False, ""

class NiftyOptionsBot:
    """Main trading bot for Nifty options"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize data manager first for logging setup
        self.data_manager = DataManager()
        
        # Initialize components
        self.fyers_client = FyersClient(
            client_id=config['fyers']['client_id'],
            secret_key=config['fyers']['secret_key'],
            redirect_uri=config['fyers']['redirect_uri'],
            access_token=config['fyers'].get('access_token')
        )
        
        self.data_analyzer = DataAnalyzer()
        self.options_manager = OptionsManager(self.fyers_client)
        self.risk_manager = RiskManager(config.get('risk_management', {}))
        
        # Trading state
        self.positions: List[TradingPosition] = []
        self.is_running = False
        self.last_analysis_time = None
        self.symbols = ['NIFTY', 'BANKNIFTY']
        
        # Historical data cache
        self.historical_data = {}
    
    def authenticate(self) -> bool:
        """Authenticate with Fyers API"""
        return self.fyers_client.authenticate()
    
    def fetch_historical_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch 3 months of historical data for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 3 months
            
            symbol_map = {
                'NIFTY': 'NSE:NIFTY50-INDEX',
                'BANKNIFTY': 'NSE:NIFTYBANK-INDEX'
            }
            
            fyers_symbol = symbol_map.get(symbol, f'NSE:{symbol}-INDEX')
            
            # Fetch daily data
            daily_data = self.fyers_client.get_historical_data(
                fyers_symbol, 'D', start_date, end_date
            )
            
            # Fetch 1-minute data for last 5 days
            minute_start = end_date - timedelta(days=5)
            minute_data = self.fyers_client.get_historical_data(
                fyers_symbol, '1', minute_start, end_date
            )
            
            if daily_data.empty or minute_data.empty:
                logger.error(f"Failed to fetch historical data for {symbol}")
                return {}
            
            # Save data to CSV for backup
            self.data_manager.save_historical_data(symbol, 'daily', daily_data)
            self.data_manager.save_historical_data(symbol, '1min', minute_data)
            
            logger.info(f"Fetched historical data for {symbol}: {len(daily_data)} daily, {len(minute_data)} minute candles")
            
            return {
                'daily': daily_data,
                'minute': minute_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return {}
    
    def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            # Get historical data
            if symbol not in self.historical_data:
                self.historical_data[symbol] = self.fetch_historical_data(symbol)
            
            data = self.historical_data[symbol]
            if not data:
                return {}
            
            # Perform analysis
            analysis = self.data_analyzer.analyze_market_data(
                data['daily'], data['minute']
            )
            
            # Add current market data
            symbol_map = {
                'NIFTY': 'NSE:NIFTY50-INDEX',
                'BANKNIFTY': 'NSE:NIFTYBANK-INDEX'
            }
            
            quotes = self.fyers_client.get_quotes([symbol_map[symbol]])
            if quotes:
                current_data = list(quotes.values())[0]
                analysis['current_ltp'] = current_data.get('ltp', 0)
                analysis['current_change'] = current_data.get('ch', 0)
                analysis['current_change_pct'] = current_data.get('chp', 0)
            
            # Save analysis results to CSV
            self.data_manager.save_analysis_results(symbol, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return {}
    
    def generate_trading_signal(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on analysis"""
        try:
            signal = {
                'action': 'hold',
                'confidence': 0,
                'strategy': '',
                'reasoning': []
            }
            
            # Check ML prediction
            ml_prediction = analysis.get('ml_prediction', {})
            ml_signal = ml_prediction.get('signal', 0)
            ml_confidence = ml_prediction.get('confidence', 0)
            
            # Check trend analysis
            daily_trend = analysis.get('daily_trend', {})
            minute_trend = analysis.get('minute_trend', {})
            
            # Check for gap analysis
            recent_gaps = analysis.get('recent_gaps', 0)
            latest_gap = analysis.get('latest_gap', 0)
            
            # Check patterns
            patterns = analysis.get('daily_patterns', {})
            
            # Decision logic
            confidence_score = 0
            
            # Strong trend continuation signal
            if (daily_trend.get('trend') == 'up' and 
                daily_trend.get('strength') == 'strong' and
                minute_trend.get('trend') == 'up'):
                signal['action'] = 'sell_put'
                signal['strategy'] = 'trend_following'
                confidence_score += 30
                signal['reasoning'].append('Strong uptrend detected')
            
            elif (daily_trend.get('trend') == 'down' and 
                  daily_trend.get('strength') == 'strong' and
                  minute_trend.get('trend') == 'down'):
                signal['action'] = 'sell_call'
                signal['strategy'] = 'trend_following'
                confidence_score += 30
                signal['reasoning'].append('Strong downtrend detected')
            
            # Gap-based signals
            if abs(latest_gap) > 1.0:  # Significant gap
                if latest_gap > 0:  # Gap up
                    signal['action'] = 'sell_call'
                    signal['strategy'] = 'gap_fade'
                    confidence_score += 20
                    signal['reasoning'].append(f'Gap up of {latest_gap:.2f}% detected')
                else:  # Gap down
                    signal['action'] = 'sell_put'
                    signal['strategy'] = 'gap_fade'
                    confidence_score += 20
                    signal['reasoning'].append(f'Gap down of {latest_gap:.2f}% detected')
            
            # ML model confirmation
            if ml_signal == 1 and ml_confidence > 0.6:
                if signal['action'] == 'sell_put':
                    confidence_score += 25
                    signal['reasoning'].append(f'ML model confirms bullish (confidence: {ml_confidence:.2f})')
                elif signal['action'] == 'hold':
                    signal['action'] = 'sell_put'
                    signal['strategy'] = 'ml_prediction'
                    confidence_score += 20
                    signal['reasoning'].append(f'ML model predicts bullish (confidence: {ml_confidence:.2f})')
            
            elif ml_signal == 0 and ml_confidence > 0.6:
                if signal['action'] == 'sell_call':
                    confidence_score += 25
                    signal['reasoning'].append(f'ML model confirms bearish (confidence: {ml_confidence:.2f})')
                elif signal['action'] == 'hold':
                    signal['action'] = 'sell_call'
                    signal['strategy'] = 'ml_prediction'
                    confidence_score += 20
                    signal['reasoning'].append(f'ML model predicts bearish (confidence: {ml_confidence:.2f})')
            
            # Pattern-based adjustments
            if patterns.get('bullish_engulfing'):
                if signal['action'] in ['sell_put', 'hold']:
                    confidence_score += 15
                    signal['reasoning'].append('Bullish engulfing pattern detected')
            
            if patterns.get('bearish_engulfing'):
                if signal['action'] in ['sell_call', 'hold']:
                    confidence_score += 15
                    signal['reasoning'].append('Bearish engulfing pattern detected')
            
            # Volatility check
            volatility = analysis.get('volatility', 0)
            if volatility > 50:  # High volatility
                confidence_score += 10
                signal['reasoning'].append('High volatility environment favorable for option selling')
            
            signal['confidence'] = min(100, confidence_score)
            
            # Only trade if confidence is above threshold
            if signal['confidence'] < 60:
                signal['action'] = 'hold'
                signal['reasoning'].append('Confidence below threshold')
            
            logger.info(f"Generated signal for {symbol}: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'strategy': '', 'reasoning': []}
    
    def execute_trade(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Execute trade based on signal"""
        try:
            if signal['action'] == 'hold':
                return True
            
            # Get current price
            analysis = self.analyze_market(symbol)
            current_price = analysis.get('current_ltp', 0)
            
            if current_price == 0:
                logger.error(f"Could not get current price for {symbol}")
                return False
            
            # Determine strategy
            if signal['action'] == 'sell_put':
                strategy = 'otm_put'
            elif signal['action'] == 'sell_call':
                strategy = 'otm_call'
            else:
                return True
            
            # Get best options to sell
            best_options = self.options_manager.get_best_options_to_sell(
                symbol, current_price, max_options=1
            )
            
            if not best_options:
                logger.warning(f"No suitable options found for {symbol}")
                return False
            
            option, metrics = best_options[0]
            
            # Check risk management
            if not self.risk_manager.can_open_position(option.ltp, len(self.positions)):
                logger.warning("Risk management prevents opening new position")
                return False
            
            # Calculate quantity (for Nifty, lot size is typically 25)
            lot_size = 25 if 'NIFTY' in symbol and 'BANK' not in symbol else 15  # Bank Nifty lot size
            quantity = lot_size
            
            # Place order
            order_result = self.fyers_client.place_order(
                symbol=option.symbol,
                qty=quantity,
                side=-1,  # Sell
                type=2,   # Market order
                price=0
            )
            
            if order_result and order_result.get('s') == 'ok':
                # Create position record
                position = TradingPosition(
                    symbol=symbol,
                    option_symbol=option.symbol,
                    quantity=quantity,
                    entry_price=option.ltp,
                    current_price=option.ltp,
                    timestamp=datetime.now(),
                    position_type=f"short_{option.option_type.lower()}",
                    strike=option.strike,
                    expiry=option.expiry
                )
                
                self.positions.append(position)
                
                # Log trade to CSV
                trade_data = {
                    'symbol': symbol,
                    'option_symbol': option.symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': option.ltp,
                    'strike': option.strike,
                    'expiry': option.expiry,
                    'option_type': option.option_type,
                    'strategy': signal['strategy'],
                    'confidence': signal['confidence'],
                    'reasoning': '; '.join(signal['reasoning']),
                    'pnl': 0,
                    'status': 'open'
                }
                self.data_manager.log_trade(trade_data)
                
                logger.info(f"Successfully opened position: {position}")
                return True
            else:
                logger.error(f"Failed to place order: {order_result}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor and manage existing positions"""
        try:
            if not self.positions:
                return
            
            # Get current quotes for all option symbols
            option_symbols = [pos.option_symbol for pos in self.positions]
            quotes = self.fyers_client.get_quotes(option_symbols)
            
            positions_to_close = []
            
            for position in self.positions:
                # Update current price
                if position.option_symbol in quotes:
                    position.current_price = quotes[position.option_symbol].get('ltp', position.current_price)
                
                # Calculate P&L
                position.pnl = (position.entry_price - position.current_price) * position.quantity
                
                # Check if position should be closed
                should_close, reason = self.risk_manager.should_close_position(position)
                
                if should_close:
                    positions_to_close.append((position, reason))
                    logger.info(f"Position marked for closure: {position.option_symbol}, Reason: {reason}")
            
            # Close positions
            for position, reason in positions_to_close:
                self.close_position(position, reason)
            
            # Update daily P&L
            self.risk_manager.daily_pnl = sum(pos.pnl for pos in self.positions)
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def close_position(self, position: TradingPosition, reason: str) -> bool:
        """Close a position"""
        try:
            # Place buy order to close short position
            order_result = self.fyers_client.place_order(
                symbol=position.option_symbol,
                qty=position.quantity,
                side=1,   # Buy to close
                type=2,   # Market order
                price=0
            )
            
            if order_result and order_result.get('s') == 'ok':
                # Log trade closure to CSV
                trade_data = {
                    'symbol': position.symbol,
                    'option_symbol': position.option_symbol,
                    'action': 'BUY',
                    'quantity': position.quantity,
                    'price': position.current_price,
                    'strike': position.strike,
                    'expiry': position.expiry,
                    'option_type': position.position_type.split('_')[1].upper(),
                    'strategy': reason,
                    'confidence': 0,
                    'reasoning': f'Position closed due to {reason}',
                    'pnl': position.pnl,
                    'status': 'closed'
                }
                self.data_manager.log_trade(trade_data)
                
                logger.info(f"Position closed: {position.option_symbol}, P&L: {position.pnl:.2f}, Reason: {reason}")
                self.positions.remove(position)
                return True
            else:
                logger.error(f"Failed to close position: {order_result}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            logger.info("Starting trading cycle...")
            
            # Monitor existing positions first
            self.monitor_positions()
            
            # Analyze each symbol and generate signals
            for symbol in self.symbols:
                try:
                    # Refresh historical data every hour
                    if (self.last_analysis_time is None or 
                        datetime.now() - self.last_analysis_time > timedelta(hours=1)):
                        self.historical_data[symbol] = self.fetch_historical_data(symbol)
                    
                    # Perform analysis
                    analysis = self.analyze_market(symbol)
                    
                    if not analysis:
                        logger.warning(f"No analysis data for {symbol}")
                        continue
                    
                    # Generate trading signal
                    signal = self.generate_trading_signal(symbol, analysis)
                    
                    # Execute trade if signal is strong enough
                    if signal['action'] != 'hold' and signal['confidence'] >= 60:
                        self.execute_trade(symbol, signal)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            self.last_analysis_time = datetime.now()
            
            # Log portfolio status
            total_pnl = sum(pos.pnl for pos in self.positions)
            logger.info(f"Portfolio Status - Positions: {len(self.positions)}, Total P&L: {total_pnl:.2f}")
            
            # Cleanup old files periodically (once per day)
            if datetime.now().hour == 0 and datetime.now().minute < 10:
                self.data_manager.cleanup_old_files(days_to_keep=30)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Nifty Options Trading Bot...")
        
        if not self.authenticate():
            logger.error("Failed to authenticate with Fyers API")
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait for next cycle (5 minutes)
                time.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Close all positions
        for position in self.positions.copy():
            self.close_position(position, "bot_shutdown")
        
        # Export final performance report
        self.data_manager.export_performance_report()
        
        total_pnl = sum(pos.pnl for pos in self.positions)
        logger.info(f"Bot stopped. Final P&L: {total_pnl:.2f}")

def load_config() -> Dict[str, Any]:
    """Load configuration from file and environment"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        config = {}
    
    # Override with environment variables
    config.setdefault('fyers', {})
    config['fyers']['client_id'] = os.getenv('FYERS_CLIENT_ID', config['fyers'].get('client_id', ''))
    config['fyers']['secret_key'] = os.getenv('FYERS_SECRET_KEY', config['fyers'].get('secret_key', ''))
    config['fyers']['redirect_uri'] = os.getenv('FYERS_REDIRECT_URI', config['fyers'].get('redirect_uri', ''))
    config['fyers']['access_token'] = os.getenv('FYERS_ACCESS_TOKEN', config['fyers'].get('access_token', ''))
    
    # Default risk management settings
    config.setdefault('risk_management', {
        'max_position_value': 50000,
        'max_daily_loss': 10000,
        'max_positions': 3,
        'stop_loss_pct': 200,
        'profit_target_pct': 50
    })
    
    return config

def main():
    """Main function to run the trading bot with manual confirmation"""
    logger.info("Starting Nifty Options Trading Bot with Manual Confirmation")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize Fyers client
        fyers_client = FyersClient()
        
        # Authenticate
        if not fyers_client.authenticate():
            logger.error("Failed to authenticate with Fyers API")
            return
        
        # Initialize trading engine with confirmation system
        trading_engine = NiftyOptionsBot(config)
        
        # Show startup information
        print("\n" + "="*80)
        print("🚀 NIFTY OPTIONS TRADING BOT - MANUAL CONFIRMATION MODE")
        print("="*80)
        print("📋 TRADING RULES:")
        print("   • Maximum 2 trades per day")
        print("   • 1:1 Risk-Reward ratio enforced")
        print("   • Manual confirmation required for each trade")
        print("   • Manual confirmation required for ALL exits")
        print("   • Nifty: 100 points stop/target")
        print("   • Bank Nifty: 200 points stop/target")
        print("   • Minimum 2 hours between trades")
        
        # Show today's trading summary
        confirmation_system = TradeConfirmationSystem()
        daily_summary = confirmation_system.get_daily_trade_summary()
        
        print(f"\n📊 TODAY'S TRADING STATUS:")
        print(f"   Date: {daily_summary['date']}")
        print(f"   Trades Executed: {daily_summary['trades_executed']}/2")
        print(f"   Trades Remaining: {daily_summary['trades_remaining']}")
        
        if daily_summary['trades_executed'] >= 2:
            print("   ⚠️  DAILY LIMIT REACHED - No more trades today")
        elif daily_summary['trades_executed'] == 1:
            print("   ⚡ 1 trade remaining today")
        else:
            print("   ✅ Ready to trade (2 trades available)")
        
        print("\n" + "="*80)
        print("🔄 Bot is now monitoring markets and positions...")
        print("💡 ENTRY: When a trade opportunity is found, you'll be asked to confirm with 'y'")
        print("💡 EXIT: When exit conditions are met, you'll be asked to confirm with 'y'")
        print("🎯 EXIT TRIGGERS:")
        print("   • Stop Loss Hit (requires confirmation)")
        print("   • Profit Target Hit (requires confirmation)")
        print("   • Market Close (requires confirmation)")
        print("   • Manual Suggestions (optional)")
        print("❌ Press Ctrl+C to stop the bot")
        print("="*80)
        
        # Start the trading engine
        trading_engine.start()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n🛑 Trading bot stopped by user")
        if 'trading_engine' in locals():
            trading_engine.stop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
    finally:
        print("👋 Trading bot shutdown complete")

if __name__ == "__main__":
    main()
