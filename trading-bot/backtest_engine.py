#!/usr/bin/env python3
"""
Backtesting Engine for Trading Bot
Tests strategies on historical data with detailed logging
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from data_analyzer import DataAnalyzer
from options_manager import OptionsManager
from data_manager import DataManager

logger = logging.getLogger(__name__)

@dataclass
class BacktestPosition:
    """Represents a position during backtesting"""
    symbol: str
    option_symbol: str
    strike: float
    expiry: datetime
    option_type: str
    entry_date: datetime
    entry_price: float
    quantity: int
    strategy: str
    confidence: float
    reasoning: str
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    max_profit: float = 0.0
    max_loss: float = 0.0
    pnl: float = 0.0

class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.data_analyzer = DataAnalyzer()
        self.positions: List[BacktestPosition] = []
        self.closed_positions: List[BacktestPosition] = []
        self.backtest_log_file = ""
        
        # Backtest settings
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.max_positions = 3
        self.commission_per_trade = 20
        
        # Risk management
        self.stop_loss_pct = 200  # 200% of premium
        self.profit_target_pct = 50  # 50% of premium
        self.max_daily_loss = 10000
        
        # Statistics
        self.daily_pnl = {}
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load historical data for backtesting"""
        try:
            # Try to load from saved files first
            historical_dir = self.data_manager.base_dir / "historical"
            
            daily_file = historical_dir / f"{symbol}_daily_latest.csv"
            minute_file = historical_dir / f"{symbol}_1min_latest.csv"
            
            data = {}
            
            if daily_file.exists():
                daily_df = pd.read_csv(daily_file, index_col=0, parse_dates=True)
                daily_df = daily_df[(daily_df.index >= start_date) & (daily_df.index <= end_date)]
                data['daily'] = daily_df
                logger.info(f"Loaded daily data for {symbol}: {len(daily_df)} records")
            
            if minute_file.exists():
                minute_df = pd.read_csv(minute_file, index_col=0, parse_dates=True)
                minute_df = minute_df[(minute_df.index >= start_date) & (minute_df.index <= end_date)]
                data['minute'] = minute_df
                logger.info(f"Loaded minute data for {symbol}: {len(minute_df)} records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return {}
    
    def simulate_options_data(self, underlying_price: float, date: datetime, symbol: str) -> List[Dict]:
        """Simulate options data for backtesting (simplified)"""
        try:
            options = []
            
            # Determine strike interval
            strike_interval = 50 if 'NIFTY' in symbol and 'BANK' not in symbol else 100
            
            # Generate strikes around current price
            base_strike = round(underlying_price / strike_interval) * strike_interval
            
            for i in range(-5, 6):  # 11 strikes total
                strike = base_strike + (i * strike_interval)
                
                # Calculate days to expiry (assume weekly expiry on Thursday)
                days_to_expiry = 7 - date.weekday() if date.weekday() < 4 else 14 - date.weekday()
                expiry = date + timedelta(days=days_to_expiry)
                
                # Simplified option pricing (for backtesting only)
                # In reality, you'd use Black-Scholes or market data
                
                # Call option
                if strike > underlying_price:  # OTM call
                    distance = (strike - underlying_price) / underlying_price
                    call_price = max(5, 50 * np.exp(-distance * 10) * (days_to_expiry / 7))
                else:  # ITM call
                    intrinsic = underlying_price - strike
                    time_value = max(5, 30 * (days_to_expiry / 7))
                    call_price = intrinsic + time_value
                
                # Put option
                if strike < underlying_price:  # OTM put
                    distance = (underlying_price - strike) / underlying_price
                    put_price = max(5, 50 * np.exp(-distance * 10) * (days_to_expiry / 7))
                else:  # ITM put
                    intrinsic = strike - underlying_price
                    time_value = max(5, 30 * (days_to_expiry / 7))
                    put_price = intrinsic + time_value
                
                # Add some randomness for realism
                call_price *= np.random.uniform(0.9, 1.1)
                put_price *= np.random.uniform(0.9, 1.1)
                
                options.append({
                    'strike': strike,
                    'expiry': expiry,
                    'call_price': call_price,
                    'put_price': put_price,
                    'volume': np.random.randint(100, 1000),
                    'oi': np.random.randint(500, 5000)
                })
            
            return options
            
        except Exception as e:
            logger.error(f"Error simulating options data: {e}")
            return []
    
    def generate_trading_signal(self, symbol: str, analysis: Dict[str, Any], date: datetime) -> Dict[str, Any]:
        """Generate trading signal for backtesting (simplified version)"""
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
            
            # Check for gap analysis
            latest_gap = analysis.get('latest_gap', 0)
            
            confidence_score = 0
            
            # Strong trend signals
            if (daily_trend.get('trend') == 'up' and 
                daily_trend.get('strength') == 'strong'):
                signal['action'] = 'sell_put'
                signal['strategy'] = 'trend_following'
                confidence_score += 40
                signal['reasoning'].append('Strong uptrend detected')
            
            elif (daily_trend.get('trend') == 'down' and 
                  daily_trend.get('strength') == 'strong'):
                signal['action'] = 'sell_call'
                signal['strategy'] = 'trend_following'
                confidence_score += 40
                signal['reasoning'].append('Strong downtrend detected')
            
            # Gap-based signals
            if abs(latest_gap) > 1.0:
                if latest_gap > 0:  # Gap up
                    signal['action'] = 'sell_call'
                    signal['strategy'] = 'gap_fade'
                    confidence_score += 30
                    signal['reasoning'].append(f'Gap up of {latest_gap:.2f}%')
                else:  # Gap down
                    signal['action'] = 'sell_put'
                    signal['strategy'] = 'gap_fade'
                    confidence_score += 30
                    signal['reasoning'].append(f'Gap down of {latest_gap:.2f}%')
            
            # ML confirmation
            if ml_signal == 1 and ml_confidence > 0.6:
                if signal['action'] == 'sell_put':
                    confidence_score += 20
                elif signal['action'] == 'hold':
                    signal['action'] = 'sell_put'
                    signal['strategy'] = 'ml_prediction'
                    confidence_score += 25
            
            elif ml_signal == 0 and ml_confidence > 0.6:
                if signal['action'] == 'sell_call':
                    confidence_score += 20
                elif signal['action'] == 'hold':
                    signal['action'] = 'sell_call'
                    signal['strategy'] = 'ml_prediction'
                    confidence_score += 25
            
            signal['confidence'] = min(100, confidence_score)
            
            # Only trade if confidence is above threshold
            if signal['confidence'] < 60:
                signal['action'] = 'hold'
                signal['reasoning'].append('Confidence below threshold')
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'strategy': '', 'reasoning': []}
    
    def execute_backtest_trade(self, symbol: str, signal: Dict[str, Any], 
                              current_price: float, date: datetime) -> bool:
        """Execute a trade during backtesting"""
        try:
            if signal['action'] == 'hold' or len(self.positions) >= self.max_positions:
                return False
            
            # Get simulated options data
            options_data = self.simulate_options_data(current_price, date, symbol)
            
            if not options_data:
                return False
            
            # Select appropriate option based on signal
            selected_option = None
            
            if signal['action'] == 'sell_put':
                # Find OTM put
                otm_puts = [opt for opt in options_data if opt['strike'] < current_price]
                if otm_puts:
                    # Select highest strike OTM put
                    selected_option = max(otm_puts, key=lambda x: x['strike'])
                    option_price = selected_option['put_price']
                    option_type = 'PE'
            
            elif signal['action'] == 'sell_call':
                # Find OTM call
                otm_calls = [opt for opt in options_data if opt['strike'] > current_price]
                if otm_calls:
                    # Select lowest strike OTM call
                    selected_option = min(otm_calls, key=lambda x: x['strike'])
                    option_price = selected_option['call_price']
                    option_type = 'CE'
            
            if not selected_option:
                return False
            
            # Create position
            lot_size = 25 if 'NIFTY' in symbol and 'BANK' not in symbol else 15
            
            position = BacktestPosition(
                symbol=symbol,
                option_symbol=f"{symbol}{selected_option['expiry'].strftime('%y%m%d')}{selected_option['strike']}{option_type}",
                strike=selected_option['strike'],
                expiry=selected_option['expiry'],
                option_type=option_type,
                entry_date=date,
                entry_price=option_price,
                quantity=lot_size,
                strategy=signal['strategy'],
                confidence=signal['confidence'],
                reasoning='; '.join(signal['reasoning'])
            )
            
            self.positions.append(position)
            self.trade_count += 1
            
            # Deduct commission
            self.current_capital -= self.commission_per_trade
            
            logger.info(f"Backtest trade executed: {position.option_symbol} at {option_price}")
            
            # Log to backtest file
            if self.backtest_log_file:
                trade_data = {
                    'timestamp': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'option_symbol': position.option_symbol,
                    'strike': position.strike,
                    'expiry': position.expiry,
                    'option_type': option_type,
                    'quantity': lot_size,
                    'entry_price': option_price,
                    'exit_price': 0,
                    'pnl': 0,
                    'strategy': signal['strategy'],
                    'confidence': signal['confidence'],
                    'reasoning': position.reasoning,
                    'days_held': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'exit_reason': ''
                }
                self.data_manager.log_backtest_trade(self.backtest_log_file, trade_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            return False
    
    def update_positions(self, symbol: str, current_price: float, date: datetime):
        """Update position values and check for exits"""
        try:
            positions_to_close = []
            
            for position in self.positions:
                if position.symbol != symbol:
                    continue
                
                # Simulate current option price
                options_data = self.simulate_options_data(current_price, date, symbol)
                current_option_price = 0
                
                for opt in options_data:
                    if opt['strike'] == position.strike:
                        if position.option_type == 'CE':
                            current_option_price = opt['call_price']
                        else:
                            current_option_price = opt['put_price']
                        break
                
                if current_option_price == 0:
                    continue
                
                # Calculate P&L (for short positions, profit when option price decreases)
                pnl = (position.entry_price - current_option_price) * position.quantity
                position.pnl = pnl
                
                # Update max profit/loss
                position.max_profit = max(position.max_profit, pnl)
                position.max_loss = min(position.max_loss, pnl)
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # Stop loss (200% of premium)
                loss_pct = (current_option_price - position.entry_price) / position.entry_price * 100
                if loss_pct > self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Profit target (50% of premium)
                profit_pct = (position.entry_price - current_option_price) / position.entry_price * 100
                if profit_pct > self.profit_target_pct:
                    should_exit = True
                    exit_reason = "profit_target"
                
                # Expiry check
                days_to_expiry = (position.expiry - date).days
                if days_to_expiry <= 0:
                    should_exit = True
                    exit_reason = "expiry"
                
                if should_exit:
                    position.exit_date = date
                    position.exit_price = current_option_price
                    position.exit_reason = exit_reason
                    positions_to_close.append(position)
            
            # Close positions
            for position in positions_to_close:
                self.close_backtest_position(position)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def close_backtest_position(self, position: BacktestPosition):
        """Close a position during backtesting"""
        try:
            # Calculate final P&L
            final_pnl = (position.entry_price - position.exit_price) * position.quantity
            position.pnl = final_pnl
            
            # Update capital
            self.current_capital += final_pnl - self.commission_per_trade
            
            # Update statistics
            if final_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Move to closed positions
            self.positions.remove(position)
            self.closed_positions.append(position)
            
            # Update daily P&L
            date_str = position.exit_date.strftime('%Y-%m-%d')
            if date_str not in self.daily_pnl:
                self.daily_pnl[date_str] = 0
            self.daily_pnl[date_str] += final_pnl
            
            logger.info(f"Closed position: {position.option_symbol}, P&L: {final_pnl:.2f}, Reason: {position.exit_reason}")
            
            # Log to backtest file
            if self.backtest_log_file:
                days_held = (position.exit_date - position.entry_date).days
                trade_data = {
                    'timestamp': position.exit_date,
                    'symbol': position.symbol,
                    'action': 'BUY',
                    'option_symbol': position.option_symbol,
                    'strike': position.strike,
                    'expiry': position.expiry,
                    'option_type': position.option_type,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'pnl': final_pnl,
                    'strategy': position.strategy,
                    'confidence': position.confidence,
                    'reasoning': position.reasoning,
                    'days_held': days_held,
                    'max_profit': position.max_profit,
                    'max_loss': position.max_loss,
                    'exit_reason': position.exit_reason
                }
                self.data_manager.log_backtest_trade(self.backtest_log_file, trade_data)
            
        except Exception as e:
            logger.error(f"Error closing backtest position: {e}")
    
    def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime, 
                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete backtest for a symbol"""
        try:
            # Initialize backtest
            backtest_id = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            self.backtest_log_file = self.data_manager.create_backtest_log(backtest_id, config)
            
            # Load historical data
            data = self.load_historical_data(symbol, start_date, end_date)
            
            if not data or 'daily' not in data:
                logger.error(f"No historical data available for {symbol}")
                return {}
            
            daily_data = data['daily']
            minute_data = data.get('minute', pd.DataFrame())
            
            # Train ML model on initial data
            if len(daily_data) > 100:
                initial_data = daily_data.iloc[:100]  # Use first 100 days for training
                self.data_analyzer.train_model(initial_data)
                test_data = daily_data.iloc[100:]  # Test on remaining data
            else:
                test_data = daily_data
            
            logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
            logger.info(f"Test data: {len(test_data)} days")
            
            # Run backtest day by day
            for date, row in test_data.iterrows():
                try:
                    current_price = row['close']
                    
                    # Get data up to current date for analysis
                    analysis_data = daily_data[daily_data.index <= date]
                    
                    if len(analysis_data) < 50:  # Need sufficient data for analysis
                        continue
                    
                    # Perform analysis
                    analysis_data_with_indicators = self.data_analyzer.calculate_technical_indicators(analysis_data)
                    analysis_data_with_gaps = self.data_analyzer.detect_gaps(analysis_data_with_indicators)
                    analysis_data_with_patterns = self.data_analyzer.analyze_candle_patterns(analysis_data_with_gaps)
                    
                    # Create analysis dict
                    analysis = {
                        'current_price': current_price,
                        'daily_trend': self.data_analyzer.identify_trend(analysis_data_with_indicators),
                        'recent_gaps': len(analysis_data_with_gaps[analysis_data_with_gaps['significant_gap']].tail(5)),
                        'latest_gap': analysis_data_with_gaps['gap_size'].iloc[-1] if len(analysis_data_with_gaps) > 0 else 0,
                        'ml_prediction': self.data_analyzer.predict_signal(analysis_data_with_indicators)
                    }
                    
                    # Update existing positions
                    self.update_positions(symbol, current_price, date)
                    
                    # Generate trading signal
                    signal = self.generate_trading_signal(symbol, analysis, date)
                    
                    # Execute trade if signal is strong
                    if signal['action'] != 'hold' and signal['confidence'] >= 60:
                        self.execute_backtest_trade(symbol, signal, current_price, date)
                    
                except Exception as e:
                    logger.error(f"Error processing date {date}: {e}")
                    continue
            
            # Close any remaining positions
            for position in self.positions.copy():
                position.exit_date = end_date
                position.exit_price = position.entry_price * 0.5  # Assume 50% loss if not closed
                position.exit_reason = "backtest_end"
                self.close_backtest_position(position)
            
            # Calculate results
            results = self.calculate_backtest_results()
            
            logger.info(f"Backtest completed for {symbol}")
            logger.info(f"Total trades: {self.trade_count}, Win rate: {results['win_rate']:.2f}%")
            logger.info(f"Total P&L: {results['total_pnl']:.2f}, Final capital: {self.current_capital:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def calculate_backtest_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        try:
            total_pnl = sum(pos.pnl for pos in self.closed_positions)
            
            results = {
                'total_trades': self.trade_count,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0,
                'total_pnl': total_pnl,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital * 100),
                'avg_win': np.mean([pos.pnl for pos in self.closed_positions if pos.pnl > 0]) if self.winning_trades > 0 else 0,
                'avg_loss': np.mean([pos.pnl for pos in self.closed_positions if pos.pnl < 0]) if self.losing_trades > 0 else 0,
                'max_win': max([pos.pnl for pos in self.closed_positions]) if self.closed_positions else 0,
                'max_loss': min([pos.pnl for pos in self.closed_positions]) if self.closed_positions else 0,
                'profit_factor': abs(sum(pos.pnl for pos in self.closed_positions if pos.pnl > 0) / 
                                   sum(pos.pnl for pos in self.closed_positions if pos.pnl < 0)) if self.losing_trades > 0 else 0,
                'avg_days_held': np.mean([(pos.exit_date - pos.entry_date).days for pos in self.closed_positions]) if self.closed_positions else 0,
                'daily_pnl': self.daily_pnl
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {e}")
            return {}
