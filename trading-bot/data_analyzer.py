#!/usr/bin/env python3
"""
Data Analyzer for Trading Bot
Performs trend analysis, gap detection, and technical analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Analyzes market data for trends, gaps, and trading signals"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        try:
            df = df.copy()
            
            # Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            
            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def detect_gaps(self, df: pd.DataFrame, gap_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect price gaps in the data
        
        Args:
            df: DataFrame with OHLC data
            gap_threshold: Minimum gap percentage to consider significant
        """
        try:
            df = df.copy()
            df = df.sort_index()
            
            # Calculate gaps
            df['prev_close'] = df['close'].shift(1)
            df['gap_up'] = ((df['open'] - df['prev_close']) / df['prev_close'] * 100) > gap_threshold
            df['gap_down'] = ((df['prev_close'] - df['open']) / df['prev_close'] * 100) > gap_threshold
            df['gap_size'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
            
            # Mark significant gaps
            df['significant_gap'] = df['gap_up'] | df['gap_down']
            
            logger.info(f"Detected {df['significant_gap'].sum()} significant gaps")
            return df
            
        except Exception as e:
            logger.error(f"Error detecting gaps: {e}")
            return df
    
    def identify_trend(self, df: pd.DataFrame, lookback_period: int = 20) -> Dict[str, str]:
        """
        Identify current trend using multiple indicators
        
        Args:
            df: DataFrame with technical indicators
            lookback_period: Number of periods to look back for trend analysis
        """
        try:
            if len(df) < lookback_period:
                return {"trend": "unknown", "strength": "weak"}
            
            recent_data = df.tail(lookback_period)
            
            # Price trend
            price_trend = "up" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "down"
            
            # Moving average trend
            ma_trend = "up" if recent_data['sma_20'].iloc[-1] > recent_data['sma_50'].iloc[-1] else "down"
            
            # MACD trend
            macd_trend = "up" if recent_data['macd'].iloc[-1] > recent_data['macd_signal'].iloc[-1] else "down"
            
            # RSI analysis
            current_rsi = recent_data['rsi'].iloc[-1]
            rsi_condition = "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
            
            # Consensus trend
            trends = [price_trend, ma_trend, macd_trend]
            trend_consensus = max(set(trends), key=trends.count)
            
            # Trend strength
            trend_agreement = trends.count(trend_consensus) / len(trends)
            strength = "strong" if trend_agreement >= 0.67 else "weak"
            
            return {
                "trend": trend_consensus,
                "strength": strength,
                "rsi_condition": rsi_condition,
                "price_trend": price_trend,
                "ma_trend": ma_trend,
                "macd_trend": macd_trend
            }
            
        except Exception as e:
            logger.error(f"Error identifying trend: {e}")
            return {"trend": "unknown", "strength": "weak"}
    
    def analyze_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze candlestick patterns for the last few candles"""
        try:
            df = df.copy()
            
            # Doji pattern
            body_size = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            df['is_doji'] = (body_size / total_range) < 0.1
            
            # Hammer pattern
            lower_shadow = df['open'].combine(df['close'], min) - df['low']
            upper_shadow = df['high'] - df['open'].combine(df['close'], max)
            df['is_hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
            
            # Engulfing patterns
            df['prev_open'] = df['open'].shift(1)
            df['prev_close'] = df['close'].shift(1)
            df['prev_high'] = df['high'].shift(1)
            df['prev_low'] = df['low'].shift(1)
            
            # Bullish engulfing
            df['bullish_engulfing'] = (
                (df['prev_close'] < df['prev_open']) &  # Previous candle was bearish
                (df['close'] > df['open']) &  # Current candle is bullish
                (df['open'] < df['prev_close']) &  # Current open below previous close
                (df['close'] > df['prev_open'])  # Current close above previous open
            )
            
            # Bearish engulfing
            df['bearish_engulfing'] = (
                (df['prev_close'] > df['prev_open']) &  # Previous candle was bullish
                (df['close'] < df['open']) &  # Current candle is bearish
                (df['open'] > df['prev_close']) &  # Current open above previous close
                (df['close'] < df['prev_open'])  # Current close below previous open
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing candle patterns: {e}")
            return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning model"""
        try:
            # Select relevant features
            feature_columns = [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal',
                'rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'atr',
                'price_change', 'price_change_abs', 'gap_size',
                'is_doji', 'is_hammer', 'bullish_engulfing', 'bearish_engulfing'
            ]
            
            # Create features dataframe
            features_df = df[feature_columns].copy()
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # Convert boolean columns to int
            bool_columns = ['is_doji', 'is_hammer', 'bullish_engulfing', 'bearish_engulfing']
            for col in bool_columns:
                features_df[col] = features_df[col].astype(int)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def create_labels(self, df: pd.DataFrame, future_periods: int = 5) -> pd.Series:
        """Create labels for supervised learning (future price movement)"""
        try:
            # Calculate future returns
            future_returns = df['close'].shift(-future_periods) / df['close'] - 1
            
            # Create labels: 1 for up, 0 for down
            labels = (future_returns > 0.005).astype(int)  # 0.5% threshold
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """Train the machine learning model"""
        try:
            # Prepare features and labels
            features_df = self.prepare_features(df)
            labels = self.create_labels(df)
            
            if features_df.empty or labels.empty:
                logger.error("Failed to prepare features or labels")
                return False
            
            # Remove rows with NaN labels
            valid_indices = ~labels.isna()
            features_df = features_df[valid_indices]
            labels = labels[valid_indices]
            
            if len(features_df) < 100:
                logger.warning("Insufficient data for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_signal(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict trading signal for the latest data"""
        try:
            if not self.is_trained:
                logger.warning("Model not trained yet")
                return {"signal": 0, "confidence": 0}
            
            # Prepare features for the latest data point
            features_df = self.prepare_features(df)
            
            if features_df.empty:
                return {"signal": 0, "confidence": 0}
            
            # Get latest features
            latest_features = features_df.iloc[-1:].values
            
            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Predict
            prediction = self.model.predict(latest_features_scaled)[0]
            confidence = self.model.predict_proba(latest_features_scaled)[0].max()
            
            return {"signal": int(prediction), "confidence": float(confidence)}
            
        except Exception as e:
            logger.error(f"Error predicting signal: {e}")
            return {"signal": 0, "confidence": 0}
    
    def analyze_market_data(self, daily_df: pd.DataFrame, minute_df: pd.DataFrame) -> Dict:
        """
        Comprehensive market analysis combining daily and minute data
        
        Args:
            daily_df: Daily timeframe data
            minute_df: 1-minute timeframe data
        """
        try:
            analysis = {}
            
            # Analyze daily data
            daily_with_indicators = self.calculate_technical_indicators(daily_df)
            daily_with_gaps = self.detect_gaps(daily_with_indicators)
            daily_with_patterns = self.analyze_candle_patterns(daily_with_gaps)
            
            # Analyze minute data
            minute_with_indicators = self.calculate_technical_indicators(minute_df)
            minute_with_patterns = self.analyze_candle_patterns(minute_with_indicators)
            
            # Daily analysis
            daily_trend = self.identify_trend(daily_with_indicators)
            analysis['daily_trend'] = daily_trend
            
            # Minute analysis
            minute_trend = self.identify_trend(minute_with_indicators, lookback_period=60)
            analysis['minute_trend'] = minute_trend
            
            # Gap analysis
            recent_gaps = daily_with_gaps[daily_with_gaps['significant_gap']].tail(5)
            analysis['recent_gaps'] = len(recent_gaps)
            analysis['latest_gap'] = recent_gaps['gap_size'].iloc[-1] if len(recent_gaps) > 0 else 0
            
            # Pattern analysis
            latest_daily_patterns = {
                'doji': bool(daily_with_patterns['is_doji'].iloc[-1]),
                'hammer': bool(daily_with_patterns['is_hammer'].iloc[-1]),
                'bullish_engulfing': bool(daily_with_patterns['bullish_engulfing'].iloc[-1]),
                'bearish_engulfing': bool(daily_with_patterns['bearish_engulfing'].iloc[-1])
            }
            analysis['daily_patterns'] = latest_daily_patterns
            
            # Train model if not trained
            if not self.is_trained:
                logger.info("Training model with historical data...")
                self.train_model(daily_with_indicators)
            
            # Get prediction
            prediction = self.predict_signal(daily_with_indicators)
            analysis['ml_prediction'] = prediction
            
            # Current market state
            analysis['current_price'] = float(minute_df['close'].iloc[-1])
            analysis['daily_change'] = float((daily_df['close'].iloc[-1] / daily_df['close'].iloc[-2] - 1) * 100)
            analysis['volatility'] = float(daily_with_indicators['atr'].iloc[-1])
            
            logger.info(f"Market analysis completed - Daily trend: {daily_trend['trend']}, ML signal: {prediction['signal']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {}
