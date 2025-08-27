#!/usr/bin/env python3
"""
=============================================================================
TRADING ENGINE - The brain that makes all trading decisions
=============================================================================

PURPOSE: This file contains the core trading logic that:
- Analyzes market conditions using 16 different scenarios
- Decides when to buy/sell options based on trends and patterns
- Manages multiple trading strategies (Adaptive Learning + Machine Learning)
- Coordinates between market analysis, risk management, and order execution
- Learns from past trades to improve future decisions

KEY FUNCTIONS:
1. analyze_market_condition() - Determines current market state (bullish/bearish/sideways)
2. select_trading_strategy() - Chooses best strategy based on market conditions
3. execute_trade_decision() - Places actual buy/sell orders
4. monitor_active_positions() - Tracks open trades and P&L
5. update_learning_models() - Improves AI based on trading results

IMPORTANT SETTINGS YOU CAN EDIT:
=============================================================================
Line 85-95: MARKET CONDITIONS - Modify how bot detects market states
Line 120-130: STRATEGY SELECTION - Change which strategies to use when
Line 180-190: POSITION SIZING - Adjust how much to trade per signal
Line 220-230: EXIT RULES - Modify when to close positions
Line 280-290: LEARNING PARAMETERS - Tune AI improvement settings
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json

from api.fyers_client import FyersClient
from data.market_analyzer import DataAnalyzer
from strategies.adaptive_strategy import AdaptiveStrategy
from strategies.ml_strategy import MLStrategy
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.options_manager import OptionsManager, OptionContract
from core.trade_confirmation import TradeConfirmationSystem
from data.data_manager import DataManager
from utils.logger import setup_logger

# =============================================================================
# MARKET CONDITIONS - Line 85-95
# Edit these to change how the bot recognizes different market states
# =============================================================================

class MarketCondition(Enum):
    """16 comprehensive market conditions - EDIT: Add/modify conditions here"""
    # TRENDING MARKETS
    STRONG_BULL_TREND = "strong_bull_trend"           # EDIT: Strong upward movement
    MODERATE_BULL_TREND = "moderate_bull_trend"       # EDIT: Moderate upward movement
    WEAK_BULL_TREND = "weak_bull_trend"               # EDIT: Weak upward movement
    STRONG_BEAR_TREND = "strong_bear_trend"           # EDIT: Strong downward movement
    MODERATE_BEAR_TREND = "moderate_bear_trend"       # EDIT: Moderate downward movement
    WEAK_BEAR_TREND = "weak_bear_trend"               # EDIT: Weak downward movement
    
    # SIDEWAYS MARKETS
    TIGHT_RANGE = "tight_range"                       # EDIT: Very narrow price range
    WIDE_RANGE = "wide_range"                         # EDIT: Wide price swings
    CONSOLIDATION = "consolidation"                   # EDIT: Price consolidating
    
    # VOLATILITY CONDITIONS
    HIGH_VOLATILITY_BULL = "high_volatility_bull"     # EDIT: Volatile upward movement
    HIGH_VOLATILITY_BEAR = "high_volatility_bear"     # EDIT: Volatile downward movement
    LOW_VOLATILITY = "low_volatility"                 # EDIT: Calm, low movement
    
    # SPECIAL CONDITIONS
    GAP_UP_TREND = "gap_up_trend"                     # EDIT: Opening gaps upward
    GAP_DOWN_TREND = "gap_down_trend"                 # EDIT: Opening gaps downward
    REVERSAL_PATTERN = "reversal_pattern"             # EDIT: Trend reversal signals
    BREAKOUT_PATTERN = "breakout_pattern"             # EDIT: Price breakout signals

# =============================================================================
# STRATEGY SELECTION RULES - Line 120-130
# Edit these to change which strategy the bot uses in different conditions
# =============================================================================

STRATEGY_MAPPING = {
    # BULLISH CONDITIONS - EDIT: Change strategies for upward markets
    MarketCondition.STRONG_BULL_TREND: 'sell_otm_puts',      # Sell out-of-money puts
    MarketCondition.MODERATE_BULL_TREND: 'bull_put_spread',   # Bull put spread strategy
    MarketCondition.WEAK_BULL_TREND: 'conservative_bull',     # Conservative bullish approach
    
    # BEARISH CONDITIONS - EDIT: Change strategies for downward markets  
    MarketCondition.STRONG_BEAR_TREND: 'sell_otm_calls',     # Sell out-of-money calls
    MarketCondition.MODERATE_BEAR_TREND: 'bear_call_spread', # Bear call spread strategy
    MarketCondition.WEAK_BEAR_TREND: 'conservative_bear',    # Conservative bearish approach
    
    # SIDEWAYS CONDITIONS - EDIT: Change strategies for range-bound markets
    MarketCondition.TIGHT_RANGE: 'iron_condor',              # Iron condor strategy
    MarketCondition.WIDE_RANGE: 'short_strangle',            # Short strangle strategy
    MarketCondition.CONSOLIDATION: 'butterfly_spread',       # Butterfly spread
    
    # VOLATILITY CONDITIONS - EDIT: Change strategies for volatile markets
    MarketCondition.HIGH_VOLATILITY_BULL: 'sell_premium',    # Sell high premium options
    MarketCondition.HIGH_VOLATILITY_BEAR: 'sell_premium',    # Sell high premium options
    MarketCondition.LOW_VOLATILITY: 'buy_straddle',          # Buy straddle for movement
    
    # SPECIAL CONDITIONS - EDIT: Change strategies for special patterns
    MarketCondition.GAP_UP_TREND: 'fade_gap',                # Fade the gap strategy
    MarketCondition.GAP_DOWN_TREND: 'fade_gap',              # Fade the gap strategy
    MarketCondition.REVERSAL_PATTERN: 'reversal_play',       # Play the reversal
    MarketCondition.BREAKOUT_PATTERN: 'momentum_play'        # Follow the momentum
}

# =============================================================================
# POSITION SIZING RULES - Line 180-190
# Edit these to control how much the bot trades per signal
# =============================================================================

POSITION_SIZING_CONFIG = {
    'base_position_size': 50000,           # EDIT: Base position size in rupees
    'confidence_multiplier': {             # EDIT: Multiply position size based on confidence
        'high': 1.5,                       # High confidence = 1.5x position size
        'medium': 1.0,                     # Medium confidence = 1x position size  
        'low': 0.5                         # Low confidence = 0.5x position size
    },
    'volatility_adjustment': {             # EDIT: Adjust position size based on volatility
        'high': 0.7,                       # High volatility = 70% of normal size
        'medium': 1.0,                     # Medium volatility = 100% of normal size
        'low': 1.3                         # Low volatility = 130% of normal size
    },
    'max_position_per_symbol': 100000,     # EDIT: Maximum position size per symbol
    'max_total_exposure': 500000           # EDIT: Maximum total exposure across all positions
}

# =============================================================================
# EXIT RULES - Line 220-230  
# Edit these to change when the bot closes positions
# =============================================================================

EXIT_RULES_CONFIG = {
    'profit_targets': {                    # EDIT: When to take profits
        'quick_profit': 0.20,              # Take 20% profit quickly
        'standard_profit': 0.50,           # Take 50% profit normally
        'max_profit': 0.80                 # Take 80% profit maximum
    },
    'stop_losses': {                       # EDIT: When to cut losses
        'tight_stop': 0.15,                # Stop loss at 15% loss
        'standard_stop': 0.25,             # Stop loss at 25% loss
        'wide_stop': 0.40                  # Stop loss at 40% loss
    },
    'time_based_exits': {                  # EDIT: Time-based exit rules
        'intraday_exit': '15:00',          # Exit all intraday positions by 3 PM
        'weekly_expiry_exit': '14:30',     # Exit weekly expiry by 2:30 PM
        'monthly_expiry_exit': '14:00'     # Exit monthly expiry by 2 PM
    },
    'greek_based_exits': {                 # EDIT: Exit based on option Greeks
        'delta_threshold': 0.30,           # Exit when delta exceeds 0.30
        'theta_decay_target': 0.50,        # Target 50% theta decay
        'vega_risk_limit': 0.25            # Limit vega exposure to 0.25
    }
}

# =============================================================================
# LEARNING PARAMETERS - Line 280-290
# Edit these to tune how the AI learns and improves
# =============================================================================

LEARNING_CONFIG = {
    'learning_rate': 0.01,                 # EDIT: How fast AI learns (0.001-0.1)
    'memory_window': 100,                  # EDIT: Number of past trades to remember
    'confidence_threshold': 0.65,          # EDIT: Minimum confidence to trade (0-1)
    'adaptation_speed': 'medium',          # EDIT: 'slow', 'medium', 'fast'
    'feature_importance_update': 10,       # EDIT: Update feature importance every N trades
    'model_retrain_frequency': 50,         # EDIT: Retrain model every N trades
    'performance_lookback': 30,            # EDIT: Days to look back for performance
    'strategy_success_threshold': 0.60     # EDIT: Success rate to keep using strategy
}

# =============================================================================
# MAIN TRADING ENGINE CLASS - Advanced users can modify this
# =============================================================================

class TradingEngine:
    """Main trading engine with adaptive learning and manual confirmation"""
    
    def __init__(self, fyers_client: FyersClient, config: Dict[str, Any]):
        self.fyers = fyers_client
        self.config = config
        self.data_manager = DataManager(fyers_client)
        self.market_analyzer = DataAnalyzer()
        self.risk_manager = RiskManager()
        self.options_manager = OptionsManager(fyers_client)
        self.position_manager = PositionManager(fyers_client)
        self.trade_confirmation = TradeConfirmationSystem()  # NEW: Add confirmation system
        
        # Initialize learning system
        self.learning_system = AdaptiveStrategy()
        self.ml_strategy = MLStrategy()
        
        # Trading state
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        
        logger.info("Trading Engine initialized with manual confirmation system")

    def authenticate(self) -> bool:
        """Authenticate with Fyers API"""
        try:
            return self.fyers.authenticate()
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def identify_market_condition(self, symbol: str, analysis: Dict[str, Any]) -> MarketCondition:
        """Identify current market condition from 16 possible states"""
        try:
            # Extract key metrics
            trend = analysis.get('daily_trend', {})
            volatility = analysis.get('volatility', 0)
            gap = analysis.get('latest_gap', 0)
            volume = analysis.get('volume_analysis', {})
            patterns = analysis.get('daily_patterns', {})
            support_resistance = analysis.get('support_resistance', {})
            
            # Check for expiry day
            current_time = datetime.now()
            if current_time.weekday() == 3:  # Thursday (weekly expiry)
                return MarketCondition.EXPIRY_DAY
            
            # Check for gaps
            if abs(gap) > 1.0:
                return MarketCondition.GAP_UP if gap > 0 else MarketCondition.GAP_DOWN
            
            # Check volatility conditions
            if volatility > 30:
                return MarketCondition.HIGH_VOLATILITY
            elif volatility < 10:
                return MarketCondition.LOW_VOLATILITY
            
            # Check trend conditions
            trend_direction = trend.get('trend', 'sideways')
            trend_strength = trend.get('strength', 'weak')
            
            if trend_direction == 'up':
                if trend_strength == 'strong':
                    return MarketCondition.STRONG_BULL_TREND
                else:
                    return MarketCondition.WEAK_BULL_TREND
            elif trend_direction == 'down':
                if trend_strength == 'strong':
                    return MarketCondition.STRONG_BEAR_TREND
                else:
                    return MarketCondition.WEAK_BEAR_TREND
            
            # Check for breakouts
            price_change = analysis.get('current_change_pct', 0)
            if abs(price_change) > 2.0:
                return MarketCondition.BREAKOUT_UP if price_change > 0 else MarketCondition.BREAKOUT_DOWN
            
            # Check for reversals
            if patterns.get('bullish_reversal'):
                return MarketCondition.REVERSAL_BULLISH
            elif patterns.get('bearish_reversal'):
                return MarketCondition.REVERSAL_BEARISH
            
            # Check for consolidation
            if support_resistance.get('in_range', False):
                return MarketCondition.CONSOLIDATION
            
            # Check for news-driven moves
            if volume.get('unusual_volume', False):
                return MarketCondition.NEWS_DRIVEN
            
            # Default to sideways range
            return MarketCondition.SIDEWAYS_RANGE
            
        except Exception as e:
            logger.error(f"Error identifying market condition: {e}")
            return MarketCondition.SIDEWAYS_RANGE
    
    def generate_adaptive_signal(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal with adaptive learning"""
        try:
            # Identify market condition
            market_condition = self.identify_market_condition(symbol, analysis)
            
            # Get performance data for this condition
            condition_perf = self.data_manager.get_performance_data(market_condition.value)
            
            # Generate signals from different strategies
            adaptive_signal = self.learning_system.generate_signal(symbol, analysis, market_condition)
            ml_signal = self.ml_strategy.generate_signal(symbol, analysis, market_condition)
            
            # Weight signals based on historical performance
            adaptive_weight = condition_perf.get('adaptive_weight', 0.5)
            ml_weight = condition_perf.get('ml_weight', 0.5)
            
            # Combine signals with adaptive weighting
            combined_confidence = (
                adaptive_signal.confidence * adaptive_weight +
                ml_signal.confidence * ml_weight
            ) / (adaptive_weight + ml_weight)
            
            # Select best performing strategy for this condition
            best_strategy = condition_perf.get('best_strategy', 'adaptive')
            if best_strategy == 'ml':
                primary_signal = ml_signal
                secondary_signal = adaptive_signal
            else:
                primary_signal = adaptive_signal
                secondary_signal = ml_signal
            
            # Create enhanced signal
            signal = {
                'action': primary_signal.action,
                'strategy': f"{best_strategy}_{market_condition.value}",
                'confidence': combined_confidence,
                'market_condition': market_condition,
                'reasoning': primary_signal.reasoning + [f"Market condition: {market_condition.value}"],
                'risk_level': self._calculate_risk_level(market_condition, combined_confidence),
                'position_size': self._calculate_position_size(market_condition, combined_confidence),
                'stop_loss': primary_signal.stop_loss,
                'take_profit': primary_signal.take_profit,
                'timestamp': datetime.now()
            }
            
            # Apply learning adjustments
            signal = self._apply_learning_adjustments(signal, condition_perf)
            
            logger.info(f"Generated signal for {symbol}: {signal['action']} ({signal['confidence']:.2f}% confidence)")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating adaptive signal: {e}")
            return self._create_hold_signal(market_condition)
    
    def _calculate_risk_level(self, condition: MarketCondition, confidence: float) -> str:
        """Calculate risk level based on market condition and confidence"""
        high_risk_conditions = [
            MarketCondition.HIGH_VOLATILITY,
            MarketCondition.NEWS_DRIVEN,
            MarketCondition.EXPIRY_DAY,
            MarketCondition.GAP_UP,
            MarketCondition.GAP_DOWN
        ]
        
        if condition in high_risk_conditions:
            return 'high'
        elif confidence < 60:
            return 'high'
        elif confidence > 80:
            return 'low'
        else:
            return 'medium'
    
    def _calculate_position_size(self, condition: MarketCondition, confidence: float) -> float:
        """Calculate position size based on market condition and confidence"""
        base_size = 1.0
        
        # Adjust for market condition
        if condition in [MarketCondition.STRONG_BULL_TREND, MarketCondition.STRONG_BEAR_TREND]:
            base_size *= 1.2
        elif condition in [MarketCondition.HIGH_VOLATILITY, MarketCondition.EXPIRY_DAY]:
            base_size *= 0.7
        elif condition == MarketCondition.LOW_VOLATILITY:
            base_size *= 1.1
        
        # Adjust for confidence
        confidence_multiplier = confidence / 100.0
        final_size = base_size * confidence_multiplier
        
        return max(0.1, min(2.0, final_size))  # Clamp between 0.1 and 2.0
    
    def _apply_learning_adjustments(self, signal: Dict[str, Any], condition_perf: Dict) -> Dict[str, Any]:
        """Apply learning-based adjustments to signal"""
        try:
            success_rate = condition_perf.get('success_rate', 0.5)
            
            # Adjust confidence based on historical success rate
            if success_rate > 0.7:
                signal['confidence'] *= 1.1  # Boost confidence for successful conditions
            elif success_rate < 0.4:
                signal['confidence'] *= 0.9  # Reduce confidence for poor performing conditions
            
            # Adjust position size based on performance
            if condition_perf.get('total_trades', 0) > 10:
                avg_pnl = condition_perf.get('total_pnl', 0) / condition_perf.get('total_trades', 1)
                if avg_pnl > 0:
                    signal['position_size'] *= 1.1
                else:
                    signal['position_size'] *= 0.9
            
            # Clamp values
            signal['confidence'] = max(0, min(100, signal['confidence']))
            signal['position_size'] = max(0.1, min(2.0, signal['position_size']))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying learning adjustments: {e}")
            return signal
    
    def _create_hold_signal(self, condition: MarketCondition) -> Dict[str, Any]:
        """Create a hold signal"""
        return {
            'action': 'hold',
            'strategy': 'safety',
            'confidence': 0,
            'market_condition': condition,
            'reasoning': ['Error in signal generation - holding position'],
            'risk_level': 'high',
            'position_size': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'timestamp': datetime.now()
        }
    
    def execute_signal(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Execute trading signal"""
        try:
            if signal['action'] == 'hold' or signal['confidence'] < 60:
                return True
            
            # Check risk management
            if not self.risk_manager.can_trade(signal):
                logger.warning(f"Risk management blocked trade for {symbol}")
                return False
            
            # Execute through position manager
            success = self.position_manager.execute_trade(
                symbol=symbol,
                signal=signal,
                fyers_client=self.fyers
            )
            
            if success:
                # Log trade for learning
                self._log_trade_for_learning(symbol, signal)
                
                # Save trade data
                trade_data = {
                    'symbol': symbol,
                    'action': signal['action'],
                    'strategy': signal['strategy'],
                    'market_condition': signal['market_condition'].value,
                    'confidence': signal['confidence'],
                    'position_size': signal['position_size'],
                    'risk_level': signal['risk_level'],
                    'reasoning': '; '.join(signal['reasoning']),
                    'timestamp': signal['timestamp']
                }
                self.data_manager.log_trade(trade_data)
                
            return success
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def _log_trade_for_learning(self, symbol: str, signal: Dict[str, Any]):
        """Log trade data for learning system"""
        condition = signal['market_condition'].value
        
        if condition not in self.data_manager.learning_data:
            self.data_manager.learning_data[condition] = []
        
        self.data_manager.learning_data[condition].append({
            'symbol': symbol,
            'signal': signal,
            'timestamp': datetime.now()
        })
    
    def update_performance(self, symbol: str, trade_result: Dict[str, Any]):
        """Update performance metrics for learning"""
        try:
            condition = trade_result.get('market_condition')
            if not condition:
                return
            
            perf = self.data_manager.get_performance_data(condition)
            
            # Update trade counts
            perf['total_trades'] += 1
            if trade_result.get('pnl', 0) > 0:
                perf['winning_trades'] += 1
            
            # Update PnL
            perf['total_pnl'] += trade_result.get('pnl', 0)
            
            # Update success rate
            perf['success_rate'] = perf['winning_trades'] / perf['total_trades']
            
            # Update strategy weights based on performance
            strategy = trade_result.get('strategy', '')
            if 'adaptive' in strategy and trade_result.get('pnl', 0) > 0:
                perf['adaptive_weight'] = min(1.0, perf.get('adaptive_weight', 0.5) + 0.05)
                perf['ml_weight'] = max(0.1, perf.get('ml_weight', 0.5) - 0.05)
            elif 'ml' in strategy and trade_result.get('pnl', 0) > 0:
                perf['ml_weight'] = min(1.0, perf.get('ml_weight', 0.5) + 0.05)
                perf['adaptive_weight'] = max(0.1, perf.get('adaptive_weight', 0.5) - 0.05)
            
            # Update best strategy
            if perf['success_rate'] > 0.6:
                if 'adaptive' in strategy:
                    perf['best_strategy'] = 'adaptive'
                elif 'ml' in strategy:
                    perf['best_strategy'] = 'ml'
            
            # Save performance data
            self.data_manager.save_performance_data(perf)
            
            logger.info(f"Updated performance for {condition}: {perf['success_rate']:.2f} success rate")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def execute_trade_decision(self):
        """Execute trade decision with manual confirmation"""
        try:
            # Check if we can trade today
            can_trade, limit_msg = self.trade_confirmation.can_trade_today()
            if not can_trade:
                logger.info(f"Trading blocked: {limit_msg}")
                return
            
            # Get market analysis
            market_condition = self.analyze_market_condition()
            if not market_condition:
                logger.info("No clear market condition detected")
                return
            
            # Select trading strategy
            strategy = self.select_trading_strategy()
            if not strategy:
                logger.info("No suitable strategy selected")
                return
            
            # Generate trade signal
            trade_signal = self.generate_trade_signal(market_condition, strategy)
            if not trade_signal or trade_signal['action'] == 'HOLD':
                logger.info("No trade signal generated or signal is HOLD")
                return
            
            # Create trade proposal with detailed reasoning
            trade_proposal = self.create_trade_proposal(trade_signal, market_condition, strategy)
            if not trade_proposal:
                logger.error("Failed to create trade proposal")
                return
            
            # Get manual confirmation from user
            logger.info("Requesting manual trade confirmation...")
            confirmed = self.trade_confirmation.get_user_confirmation(trade_proposal)
            
            if not confirmed:
                logger.info("Trade rejected by user or system")
                return
            
            # Execute the confirmed trade
            self.execute_confirmed_trade(trade_proposal)
            
        except Exception as e:
            logger.error(f"Error in trade execution: {e}")
    
    def create_trade_proposal(self, trade_signal: Dict[str, Any], 
                            market_condition: Dict[str, Any], 
                            strategy: Dict[str, Any]) -> Optional[Any]:
        """Create detailed trade proposal with reasoning"""
        try:
            symbol = trade_signal['symbol']
            direction = trade_signal['action']  # 'BUY' or 'SELL'
            entry_price = trade_signal['entry_price']
            
            # Calculate position size
            quantity = self.risk_manager.calculate_position_size(
                entry_price=entry_price,
                stop_loss=trade_signal.get('stop_loss', entry_price * 0.95),
                risk_amount=self.config.get('risk_per_trade', 1000)
            )
            
            # Create detailed reasoning
            reasoning = {
                'market_condition': market_condition.get('condition', 'Unknown'),
                'trend_strength': market_condition.get('trend_strength', 'Unknown'),
                'strategy_used': strategy.get('name', 'Unknown'),
                'technical_signals': market_condition.get('signals', {}),
                'confidence_score': trade_signal.get('confidence', 0.0),
                'volatility': market_condition.get('volatility', 'Unknown'),
                'volume_analysis': market_condition.get('volume_analysis', 'Unknown'),
                'gap_analysis': market_condition.get('gap_analysis', 'No gaps'),
                'support_resistance': market_condition.get('support_resistance', 'Unknown'),
                'time_of_day': datetime.now().strftime('%H:%M'),
                'days_to_expiry': trade_signal.get('days_to_expiry', 'N/A')
            }
            
            # Create trade proposal using confirmation system
            proposal = self.trade_confirmation.create_trade_proposal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                reasoning=reasoning
            )
            
            return proposal
            
        except Exception as e:
            logger.error(f"Error creating trade proposal: {e}")
            return None
    
    def execute_confirmed_trade(self, trade_proposal):
        """Execute the trade after user confirmation"""
        try:
            logger.info(f"Executing confirmed trade: {trade_proposal.trade_id}")
            
            # Validate trade one more time
            validation_result = self.risk_manager.validate_trade({
                'symbol': trade_proposal.symbol,
                'action': trade_proposal.direction,
                'quantity': trade_proposal.quantity,
                'entry_price': trade_proposal.entry_price,
                'stop_loss': trade_proposal.stop_loss,
                'profit_target': trade_proposal.profit_target
            })
            
            if not validation_result['valid']:
                logger.error(f"Trade validation failed: {validation_result['reason']}")
                return
            
            # Place the order
            order_result = self.place_order(
                symbol=trade_proposal.symbol,
                action=trade_proposal.direction,
                quantity=trade_proposal.quantity,
                price=trade_proposal.entry_price,
                order_type='LIMIT'
            )
            
            if order_result and order_result.get('status') == 'success':
                logger.info(f"✅ Trade executed successfully: {trade_proposal.trade_id}")
                
                # Add to position manager
                self.position_manager.add_position({
                    'trade_id': trade_proposal.trade_id,
                    'symbol': trade_proposal.symbol,
                    'direction': trade_proposal.direction,
                    'quantity': trade_proposal.quantity,
                    'entry_price': trade_proposal.entry_price,
                    'stop_loss': trade_proposal.stop_loss,
                    'profit_target': trade_proposal.profit_target,
                    'timestamp': trade_proposal.timestamp,
                    'order_id': order_result.get('order_id')
                })
                
                # Update trade count
                self.trade_count += 1
                
                # Show trade summary
                self.show_trade_summary(trade_proposal, order_result)
                
            else:
                logger.error(f"❌ Failed to execute trade: {order_result}")
                
        except Exception as e:
            logger.error(f"Error executing confirmed trade: {e}")
    
    def show_trade_summary(self, trade_proposal, order_result):
        """Show executed trade summary"""
        print("\n" + "="*60)
        print("✅ TRADE EXECUTED SUCCESSFULLY")
        print("="*60)
        print(f"Trade ID: {trade_proposal.trade_id}")
        print(f"Order ID: {order_result.get('order_id', 'N/A')}")
        print(f"Symbol: {trade_proposal.symbol}")
        print(f"Direction: {trade_proposal.direction}")
        print(f"Quantity: {trade_proposal.quantity}")
        print(f"Entry Price: ₹{trade_proposal.entry_price:,.2f}")
        print(f"Stop Loss: ₹{trade_proposal.stop_loss:,.2f}")
        print(f"Profit Target: ₹{trade_proposal.profit_target:,.2f}")
        print(f"Risk Amount: ₹{trade_proposal.risk_amount:,.2f}")
        print(f"Potential Reward: ₹{trade_proposal.reward_amount:,.2f}")
        
        # Show daily summary
        daily_summary = self.trade_confirmation.get_daily_trade_summary()
        print(f"\n📊 Today's Trading Summary:")
        print(f"Trades Executed: {daily_summary['trades_executed']}/2")
        print(f"Trades Remaining: {daily_summary['trades_remaining']}")
        print("="*60)

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            logger.info("Starting trading cycle...")
            
            # Monitor existing positions
            self.position_manager.monitor_positions(self.fyers)
            
            # Process each symbol
            for symbol in self.config['symbols']:
                try:
                    # Get market analysis
                    analysis = self.market_analyzer.analyze_symbol(symbol, self.fyers)
                    
                    if not analysis:
                        logger.warning(f"No analysis data for {symbol}")
                        continue
                    
                    # Generate adaptive signal
                    signal = self.generate_adaptive_signal(symbol, analysis)
                    
                    # Execute signal
                    if signal['confidence'] >= 60:
                        self.execute_signal(symbol, signal)
                    
                    # Save analysis results
                    self.data_manager.save_analysis_results(symbol, analysis)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Update learning models
            self._update_learning_models()
            
            # Log cycle completion
            positions_count = len(self.position_manager.positions)
            total_pnl = sum(pos.pnl for pos in self.position_manager.positions)
            logger.info(f"Trading cycle completed - Positions: {positions_count}, Total P&L: {total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _update_learning_models(self):
        """Update ML models with recent performance data"""
        try:
            # Update adaptive strategy
            self.learning_system.update_model(self.data_manager.get_performance_data())
            
            # Update ML strategy
            self.ml_strategy.update_model(self.data_manager.learning_data)
            
            # Save updated models
            self.data_manager.save_model_data({
                'adaptive_weights': self.learning_system.get_weights(),
                'ml_model': self.ml_strategy.get_model_state(),
                'performance_tracker': self.data_manager.get_performance_data()
            })
            
        except Exception as e:
            logger.error(f"Error updating learning models: {e}")
    
    def start(self):
        """Start the trading engine"""
        logger.info("Starting Advanced Trading Engine...")
        
        if not self.authenticate():
            logger.error("Failed to authenticate with Fyers API")
            return
        
        # Load previous performance data
        self._load_performance_data()
        
        try:
            while True:
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait for next cycle (5 minutes)
                time.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("Trading engine stopped by user")
        except Exception as e:
            logger.error(f"Trading engine error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping trading engine...")
        
        # Close all positions
        self.position_manager.close_all_positions(self.fyers, "engine_shutdown")
        
        # Save final performance data
        self.data_manager.save_performance_data(self.data_manager.get_performance_data())
        self.data_manager.export_performance_report()
        
        logger.info("Trading engine stopped successfully")
    
    def _load_performance_data(self):
        """Load previous performance data"""
        try:
            saved_data = self.data_manager.load_performance_data()
            if saved_data:
                self.data_manager.performance_data.update(saved_data)
                logger.info("Loaded previous performance data")
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")

@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence and reasoning"""
    action: str  # 'buy', 'sell', 'hold'
    strategy: str
    confidence: float
    market_condition: MarketCondition
    reasoning: List[str]
    risk_level: str
    position_size: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
