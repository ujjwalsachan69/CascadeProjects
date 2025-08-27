#!/usr/bin/env python3
"""
Adaptive Strategy with Learning for All 16 Market Conditions
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class StrategySignal:
    """Strategy signal output"""
    action: str
    confidence: float
    reasoning: List[str]
    stop_loss: float
    take_profit: float

class AdaptiveStrategy:
    """Adaptive strategy that learns from market conditions"""
    
    def __init__(self):
        self.condition_weights = {}
        self.performance_history = {}
        self.learning_rate = 0.05
        
        # Initialize weights for all 16 conditions
        conditions = [
            'strong_bull_trend', 'weak_bull_trend', 'strong_bear_trend', 'weak_bear_trend',
            'sideways_range', 'breakout_up', 'breakout_down', 'gap_up', 'gap_down',
            'high_volatility', 'low_volatility', 'reversal_bullish', 'reversal_bearish',
            'consolidation', 'news_driven', 'expiry_day'
        ]
        
        for condition in conditions:
            self.condition_weights[condition] = {
                'trend_weight': 0.3,
                'momentum_weight': 0.25,
                'volatility_weight': 0.2,
                'volume_weight': 0.15,
                'pattern_weight': 0.1
            }
    
    def generate_signal(self, symbol: str, analysis: Dict[str, Any], market_condition) -> StrategySignal:
        """Generate trading signal based on market condition"""
        try:
            condition_name = market_condition.value
            weights = self.condition_weights.get(condition_name, {})
            
            # Extract analysis components
            trend = analysis.get('daily_trend', {})
            momentum = analysis.get('momentum_indicators', {})
            volatility = analysis.get('volatility', 0)
            volume = analysis.get('volume_analysis', {})
            patterns = analysis.get('daily_patterns', {})
            
            # Calculate component scores
            trend_score = self._calculate_trend_score(trend, condition_name)
            momentum_score = self._calculate_momentum_score(momentum, condition_name)
            volatility_score = self._calculate_volatility_score(volatility, condition_name)
            volume_score = self._calculate_volume_score(volume, condition_name)
            pattern_score = self._calculate_pattern_score(patterns, condition_name)
            
            # Weighted combination
            total_score = (
                trend_score * weights.get('trend_weight', 0.3) +
                momentum_score * weights.get('momentum_weight', 0.25) +
                volatility_score * weights.get('volatility_weight', 0.2) +
                volume_score * weights.get('volume_weight', 0.15) +
                pattern_score * weights.get('pattern_weight', 0.1)
            )
            
            # Generate signal based on condition-specific logic
            signal = self._condition_specific_signal(condition_name, total_score, analysis)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating adaptive signal: {e}")
            return StrategySignal('hold', 0, ['Error in signal generation'], 0, 0)
    
    def _calculate_trend_score(self, trend: Dict, condition: str) -> float:
        """Calculate trend component score"""
        try:
            trend_direction = trend.get('trend', 'sideways')
            trend_strength = trend.get('strength', 'weak')
            
            if condition in ['strong_bull_trend', 'weak_bull_trend', 'breakout_up']:
                return 80 if trend_direction == 'up' else -20
            elif condition in ['strong_bear_trend', 'weak_bear_trend', 'breakout_down']:
                return 80 if trend_direction == 'down' else -20
            elif condition in ['reversal_bullish']:
                return 60 if trend_direction == 'down' else 20  # Expecting reversal
            elif condition in ['reversal_bearish']:
                return 60 if trend_direction == 'up' else 20   # Expecting reversal
            else:
                return 40  # Neutral for sideways/consolidation
                
        except Exception:
            return 0
    
    def _calculate_momentum_score(self, momentum: Dict, condition: str) -> float:
        """Calculate momentum component score"""
        try:
            rsi = momentum.get('rsi', 50)
            macd_signal = momentum.get('macd_signal', 0)
            
            if condition in ['strong_bull_trend', 'breakout_up']:
                return min(80, max(20, rsi - 30))  # Favor higher RSI
            elif condition in ['strong_bear_trend', 'breakout_down']:
                return min(80, max(20, 70 - rsi))  # Favor lower RSI
            elif condition == 'reversal_bullish':
                return min(80, max(20, 40 - rsi))  # Oversold conditions
            elif condition == 'reversal_bearish':
                return min(80, max(20, rsi - 60))  # Overbought conditions
            else:
                return 50  # Neutral
                
        except Exception:
            return 0
    
    def _calculate_volatility_score(self, volatility: float, condition: str) -> float:
        """Calculate volatility component score"""
        try:
            if condition in ['high_volatility', 'news_driven', 'gap_up', 'gap_down']:
                return min(80, volatility * 2)  # Favor high volatility
            elif condition in ['low_volatility', 'consolidation']:
                return max(20, 80 - volatility * 2)  # Favor low volatility
            elif condition == 'expiry_day':
                return min(70, volatility * 1.5)  # Moderate volatility preference
            else:
                return 50  # Neutral
                
        except Exception:
            return 0
    
    def _calculate_volume_score(self, volume: Dict, condition: str) -> float:
        """Calculate volume component score"""
        try:
            volume_ratio = volume.get('volume_ratio', 1.0)
            unusual_volume = volume.get('unusual_volume', False)
            
            if condition in ['breakout_up', 'breakout_down', 'news_driven']:
                return 80 if unusual_volume else 30
            elif condition in ['consolidation', 'sideways_range']:
                return 70 if volume_ratio < 1.2 else 30
            else:
                return min(80, volume_ratio * 40)
                
        except Exception:
            return 0
    
    def _calculate_pattern_score(self, patterns: Dict, condition: str) -> float:
        """Calculate pattern component score"""
        try:
            bullish_patterns = patterns.get('bullish_engulfing', False) or patterns.get('hammer', False)
            bearish_patterns = patterns.get('bearish_engulfing', False) or patterns.get('shooting_star', False)
            
            if condition in ['reversal_bullish', 'weak_bull_trend']:
                return 80 if bullish_patterns else 40
            elif condition in ['reversal_bearish', 'weak_bear_trend']:
                return 80 if bearish_patterns else 40
            else:
                return 50
                
        except Exception:
            return 0
    
    def _condition_specific_signal(self, condition: str, score: float, analysis: Dict) -> StrategySignal:
        """Generate condition-specific trading signal"""
        try:
            current_price = analysis.get('current_ltp', 100)
            
            # Condition-specific signal logic
            if condition in ['strong_bull_trend', 'weak_bull_trend', 'breakout_up', 'reversal_bullish']:
                if score > 60:
                    return StrategySignal(
                        action='sell_put',
                        confidence=min(95, score),
                        reasoning=[f'Bullish {condition} detected with score {score:.1f}'],
                        stop_loss=current_price * 0.98,
                        take_profit=current_price * 1.02
                    )
            
            elif condition in ['strong_bear_trend', 'weak_bear_trend', 'breakout_down', 'reversal_bearish']:
                if score > 60:
                    return StrategySignal(
                        action='sell_call',
                        confidence=min(95, score),
                        reasoning=[f'Bearish {condition} detected with score {score:.1f}'],
                        stop_loss=current_price * 1.02,
                        take_profit=current_price * 0.98
                    )
            
            elif condition in ['gap_up']:
                if score > 55:
                    return StrategySignal(
                        action='sell_call',
                        confidence=min(90, score),
                        reasoning=[f'Gap up fade strategy with score {score:.1f}'],
                        stop_loss=current_price * 1.015,
                        take_profit=current_price * 0.995
                    )
            
            elif condition in ['gap_down']:
                if score > 55:
                    return StrategySignal(
                        action='sell_put',
                        confidence=min(90, score),
                        reasoning=[f'Gap down fade strategy with score {score:.1f}'],
                        stop_loss=current_price * 0.985,
                        take_profit=current_price * 1.005
                    )
            
            elif condition in ['high_volatility', 'expiry_day']:
                if score > 65:
                    return StrategySignal(
                        action='sell_strangle',
                        confidence=min(85, score),
                        reasoning=[f'High volatility strategy with score {score:.1f}'],
                        stop_loss=current_price * 0.97,
                        take_profit=current_price * 1.03
                    )
            
            elif condition in ['consolidation', 'sideways_range']:
                if score > 60:
                    return StrategySignal(
                        action='sell_iron_condor',
                        confidence=min(80, score),
                        reasoning=[f'Range-bound strategy with score {score:.1f}'],
                        stop_loss=current_price * 0.98,
                        take_profit=current_price * 1.02
                    )
            
            # Default hold signal
            return StrategySignal(
                action='hold',
                confidence=max(0, score - 20),
                reasoning=[f'Score {score:.1f} below threshold for {condition}'],
                stop_loss=0,
                take_profit=0
            )
            
        except Exception as e:
            logger.error(f"Error in condition-specific signal: {e}")
            return StrategySignal('hold', 0, ['Error in signal generation'], 0, 0)
    
    def update_model(self, performance_data: Dict[str, Any]):
        """Update strategy weights based on performance"""
        try:
            for condition, perf in performance_data.items():
                if condition in self.condition_weights:
                    success_rate = perf.get('success_rate', 0.5)
                    total_trades = perf.get('total_trades', 0)
                    
                    if total_trades > 5:  # Minimum trades for learning
                        # Adjust weights based on performance
                        adjustment = (success_rate - 0.5) * self.learning_rate
                        
                        weights = self.condition_weights[condition]
                        for key in weights:
                            weights[key] = max(0.05, min(0.8, weights[key] + adjustment))
                        
                        # Normalize weights
                        total_weight = sum(weights.values())
                        for key in weights:
                            weights[key] /= total_weight
            
            logger.info("Updated adaptive strategy weights based on performance")
            
        except Exception as e:
            logger.error(f"Error updating adaptive model: {e}")
    
    def get_weights(self) -> Dict[str, Any]:
        """Get current strategy weights"""
        return self.condition_weights
