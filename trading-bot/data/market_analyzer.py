#!/usr/bin/env python3
"""
=============================================================================
MARKET ANALYZER - Analyzes market trends and generates trading signals
=============================================================================

PURPOSE: This file analyzes market data to make trading decisions:
- Downloads and processes 3 months of historical price data
- Detects market trends (bullish, bearish, sideways)
- Identifies price gaps and breakout patterns
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Combines multiple timeframes for better accuracy

KEY FUNCTIONS:
1. analyze_symbol() - Main analysis function combining daily + minute data
2. _analyze_trend() - Identifies market direction and strength
3. _detect_gaps() - Finds opening price gaps
4. _calculate_indicators() - Calculates RSI, MACD, moving averages
5. _combine_signals() - Merges analysis from different timeframes

IMPORTANT SETTINGS YOU CAN EDIT:
=============================================================================
Line 55-65: TREND DETECTION - How sensitive trend detection should be
Line 70-80: TECHNICAL INDICATORS - Which indicators to use and their periods
Line 85-95: GAP DETECTION - Minimum gap size to consider significant
Line 100-110: VOLUME ANALYSIS - Volume thresholds for confirmation
Line 115-125: PATTERN RECOGNITION - Candlestick pattern settings
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import ta

from utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketAnalyzer:
    """Enhanced market analyzer with comprehensive technical analysis"""
    
    def __init__(self):
        self.cache = {}
        
    def analyze_symbol(self, symbol: str, fyers_client) -> Dict[str, Any]:
        """Comprehensive analysis for a symbol"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            symbol_map = {
                'NIFTY': 'NSE:NIFTY50-INDEX',
                'BANKNIFTY': 'NSE:NIFTYBANK-INDEX'
            }
            
            fyers_symbol = symbol_map.get(symbol, f'NSE:{symbol}-INDEX')
            
            # Fetch data
            daily_data = fyers_client.get_historical_data(fyers_symbol, 'D', start_date, end_date)
            minute_data = fyers_client.get_historical_data(fyers_symbol, '1', end_date - timedelta(days=5), end_date)
            
            if daily_data.empty:
                return {}
            
            # Perform comprehensive analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'daily_trend': self._analyze_trend(daily_data),
                'minute_trend': self._analyze_trend(minute_data) if not minute_data.empty else {},
                'technical_indicators': self._calculate_indicators(daily_data),
                'momentum_indicators': self._calculate_momentum(daily_data),
                'volatility': self._calculate_volatility(daily_data),
                'volume_analysis': self._analyze_volume(daily_data),
                'support_resistance': self._find_support_resistance(daily_data),
                'daily_patterns': self._detect_patterns(daily_data),
                'recent_gaps': self._detect_gaps(daily_data),
                'latest_gap': self._calculate_latest_gap(daily_data),
                'current_ltp': daily_data['close'].iloc[-1],
                'current_change': daily_data['close'].iloc[-1] - daily_data['close'].iloc[-2],
                'current_change_pct': ((daily_data['close'].iloc[-1] / daily_data['close'].iloc[-2]) - 1) * 100,
                'high_low_ratio': daily_data['high'].iloc[-1] / daily_data['low'].iloc[-1]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {}
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            if len(df) < 20:
                return {'trend': 'sideways', 'strength': 'weak'}
            
            # Calculate moving averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            # Determine trend
            if current_price > sma_20 > sma_50:
                trend = 'up'
            elif current_price < sma_20 < sma_50:
                trend = 'down'
            else:
                trend = 'sideways'
            
            # Determine strength
            price_change_20 = (current_price / df['close'].iloc[-20] - 1) * 100
            strength = 'strong' if abs(price_change_20) > 5 else 'weak'
            
            return {
                'trend': trend,
                'strength': strength,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'price_vs_sma20': (current_price / sma_20 - 1) * 100
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'trend': 'sideways', 'strength': 'weak'}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
            indicators['ema_20'] = ta.trend.ema_indicator(df['close'], window=20).iloc[-1]
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(df['close']).iloc[-1]
            bb_low = ta.volatility.bollinger_lband(df['close']).iloc[-1]
            indicators['bb_upper'] = bb_high
            indicators['bb_lower'] = bb_low
            
            # ATR
            indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        try:
            momentum = {}
            
            # RSI
            momentum['rsi'] = ta.momentum.rsi(df['close']).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd(df['close']).iloc[-1]
            macd_signal = ta.trend.macd_signal(df['close']).iloc[-1]
            momentum['macd'] = macd_line
            momentum['macd_signal'] = macd_signal
            momentum['macd_histogram'] = macd_line - macd_signal
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return {}
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volatility"""
        try:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            return {
                'volume_ratio': volume_ratio,
                'unusual_volume': volume_ratio > 1.5,
                'avg_volume': avg_volume,
                'current_volume': current_volume
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {}
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find support and resistance levels"""
        try:
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            resistance = highs.iloc[-20:].max()
            support = lows.iloc[-20:].min()
            current_price = df['close'].iloc[-1]
            
            return {
                'resistance': resistance,
                'support': support,
                'in_range': support < current_price < resistance,
                'distance_to_resistance': (resistance / current_price - 1) * 100,
                'distance_to_support': (current_price / support - 1) * 100
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return {}
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detect candlestick patterns"""
        try:
            patterns = {}
            
            if len(df) < 2:
                return patterns
            
            # Get last two candles
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            
            # Bullish engulfing
            patterns['bullish_engulfing'] = (
                prev['close'] < prev['open'] and  # Previous red
                curr['close'] > curr['open'] and  # Current green
                curr['open'] < prev['close'] and  # Gap down open
                curr['close'] > prev['open']      # Engulfs previous
            )
            
            # Bearish engulfing
            patterns['bearish_engulfing'] = (
                prev['close'] > prev['open'] and  # Previous green
                curr['close'] < curr['open'] and  # Current red
                curr['open'] > prev['close'] and  # Gap up open
                curr['close'] < prev['open']      # Engulfs previous
            )
            
            # Hammer
            body_size = abs(curr['close'] - curr['open'])
            lower_shadow = curr['open'] - curr['low'] if curr['close'] > curr['open'] else curr['close'] - curr['low']
            patterns['hammer'] = lower_shadow > 2 * body_size and curr['high'] - max(curr['open'], curr['close']) < body_size
            
            # Shooting star
            upper_shadow = curr['high'] - max(curr['open'], curr['close'])
            patterns['shooting_star'] = upper_shadow > 2 * body_size and min(curr['open'], curr['close']) - curr['low'] < body_size
            
            # Reversal patterns
            patterns['bullish_reversal'] = patterns['bullish_engulfing'] or patterns['hammer']
            patterns['bearish_reversal'] = patterns['bearish_engulfing'] or patterns['shooting_star']
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    def _detect_gaps(self, df: pd.DataFrame) -> int:
        """Count recent gaps"""
        try:
            gaps = 0
            for i in range(1, min(10, len(df))):
                prev_high = df['high'].iloc[-i-1]
                curr_low = df['low'].iloc[-i]
                prev_low = df['low'].iloc[-i-1]
                curr_high = df['high'].iloc[-i]
                
                # Gap up or gap down
                if curr_low > prev_high or curr_high < prev_low:
                    gaps += 1
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting gaps: {e}")
            return 0
    
    def _calculate_latest_gap(self, df: pd.DataFrame) -> float:
        """Calculate latest gap percentage"""
        try:
            if len(df) < 2:
                return 0
            
            prev_close = df['close'].iloc[-2]
            curr_open = df['open'].iloc[-1]
            
            gap_pct = (curr_open / prev_close - 1) * 100
            return gap_pct
            
        except Exception as e:
            logger.error(f"Error calculating latest gap: {e}")
            return 0
