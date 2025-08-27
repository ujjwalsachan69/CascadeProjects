#!/usr/bin/env python3
"""
Machine Learning Strategy with Neural Network Pattern Recognition
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MLSignal:
    """ML strategy signal output"""
    action: str
    confidence: float
    reasoning: List[str]
    stop_loss: float
    take_profit: float

class MLStrategy:
    """Machine Learning strategy with adaptive neural networks"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_data = {}
        
        # Initialize models for each market condition
        self.condition_models = [
            'strong_bull_trend', 'weak_bull_trend', 'strong_bear_trend', 'weak_bear_trend',
            'sideways_range', 'breakout_up', 'breakout_down', 'gap_up', 'gap_down',
            'high_volatility', 'low_volatility', 'reversal_bullish', 'reversal_bearish',
            'consolidation', 'news_driven', 'expiry_day'
        ]
        
        for condition in self.condition_models:
            self.models[condition] = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'nn': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
            }
            self.scalers[condition] = StandardScaler()
            self.training_data[condition] = []
    
    def generate_signal(self, symbol: str, analysis: Dict[str, Any], market_condition) -> MLSignal:
        """Generate ML-based trading signal"""
        try:
            condition_name = market_condition.value
            
            # Extract features
            features = self._extract_features(analysis)
            
            if len(features) == 0:
                return MLSignal('hold', 0, ['No features available'], 0, 0)
            
            # Get model predictions
            rf_prediction, rf_confidence = self._get_model_prediction(condition_name, 'rf', features)
            nn_prediction, nn_confidence = self._get_model_prediction(condition_name, 'nn', features)
            
            # Combine predictions
            combined_prediction = (rf_prediction + nn_prediction) / 2
            combined_confidence = (rf_confidence + nn_confidence) / 2
            
            # Generate signal based on prediction
            signal = self._prediction_to_signal(
                combined_prediction, combined_confidence, condition_name, analysis
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return MLSignal('hold', 0, ['Error in ML signal generation'], 0, 0)
    
    def _extract_features(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Extract features for ML model"""
        try:
            features = []
            
            # Trend features
            trend = analysis.get('daily_trend', {})
            features.extend([
                1 if trend.get('trend') == 'up' else -1 if trend.get('trend') == 'down' else 0,
                1 if trend.get('strength') == 'strong' else 0
            ])
            
            # Technical indicators
            indicators = analysis.get('technical_indicators', {})
            features.extend([
                indicators.get('sma_20', 0),
                indicators.get('ema_20', 0),
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('bb_upper', 0),
                indicators.get('bb_lower', 0),
                indicators.get('atr', 0)
            ])
            
            # Volume features
            volume = analysis.get('volume_analysis', {})
            features.extend([
                volume.get('volume_ratio', 1.0),
                1 if volume.get('unusual_volume', False) else 0
            ])
            
            # Volatility
            features.append(analysis.get('volatility', 0))
            
            # Gap analysis
            features.append(analysis.get('latest_gap', 0))
            
            # Pattern features
            patterns = analysis.get('daily_patterns', {})
            features.extend([
                1 if patterns.get('bullish_engulfing', False) else 0,
                1 if patterns.get('bearish_engulfing', False) else 0,
                1 if patterns.get('hammer', False) else 0,
                1 if patterns.get('shooting_star', False) else 0
            ])
            
            # Price action features
            features.extend([
                analysis.get('current_change_pct', 0),
                analysis.get('high_low_ratio', 1.0)
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])
    
    def _get_model_prediction(self, condition: str, model_type: str, features: np.ndarray) -> tuple:
        """Get prediction from specific model"""
        try:
            if condition not in self.models:
                return 0.5, 0.5  # Neutral prediction
            
            model = self.models[condition][model_type]
            scaler = self.scalers[condition]
            
            # Check if model is trained
            if not hasattr(model, 'classes_'):
                return 0.5, 0.5  # Model not trained yet
            
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Get prediction
            prediction = model.predict_proba(scaled_features)[0]
            
            # Return prediction and confidence
            if len(prediction) == 2:
                confidence = max(prediction) - 0.5  # Confidence above random
                signal = 1 if prediction[1] > prediction[0] else 0
                return signal, confidence * 100
            else:
                return 0.5, 50
                
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return 0.5, 50
    
    def _prediction_to_signal(self, prediction: float, confidence: float, condition: str, analysis: Dict) -> MLSignal:
        """Convert ML prediction to trading signal"""
        try:
            current_price = analysis.get('current_ltp', 100)
            
            # Determine action based on prediction and condition
            if prediction > 0.6 and confidence > 60:
                if condition in ['strong_bull_trend', 'weak_bull_trend', 'breakout_up', 'reversal_bullish', 'gap_down']:
                    action = 'sell_put'
                    reasoning = [f'ML predicts bullish movement (confidence: {confidence:.1f}%)']
                    stop_loss = current_price * 0.98
                    take_profit = current_price * 1.02
                else:
                    action = 'sell_call'
                    reasoning = [f'ML predicts bearish movement (confidence: {confidence:.1f}%)']
                    stop_loss = current_price * 1.02
                    take_profit = current_price * 0.98
            
            elif prediction < 0.4 and confidence > 60:
                if condition in ['strong_bear_trend', 'weak_bear_trend', 'breakout_down', 'reversal_bearish', 'gap_up']:
                    action = 'sell_call'
                    reasoning = [f'ML predicts bearish movement (confidence: {confidence:.1f}%)']
                    stop_loss = current_price * 1.02
                    take_profit = current_price * 0.98
                else:
                    action = 'sell_put'
                    reasoning = [f'ML predicts bullish movement (confidence: {confidence:.1f}%)']
                    stop_loss = current_price * 0.98
                    take_profit = current_price * 1.02
            
            else:
                action = 'hold'
                reasoning = [f'ML prediction uncertain (confidence: {confidence:.1f}%)']
                stop_loss = 0
                take_profit = 0
            
            return MLSignal(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            logger.error(f"Error converting prediction to signal: {e}")
            return MLSignal('hold', 0, ['Error in signal conversion'], 0, 0)
    
    def update_model(self, learning_data: Dict[str, List]):
        """Update ML models with new data"""
        try:
            for condition, data_points in learning_data.items():
                if condition in self.models and len(data_points) > 10:
                    
                    # Prepare training data
                    X, y = self._prepare_training_data(data_points)
                    
                    if len(X) > 20:  # Minimum data for training
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Scale features
                        scaler = self.scalers[condition]
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train Random Forest
                        rf_model = self.models[condition]['rf']
                        rf_model.fit(X_train_scaled, y_train)
                        
                        # Train Neural Network
                        nn_model = self.models[condition]['nn']
                        nn_model.fit(X_train_scaled, y_train)
                        
                        # Evaluate models
                        rf_score = rf_model.score(X_test_scaled, y_test)
                        nn_score = nn_model.score(X_test_scaled, y_test)
                        
                        logger.info(f"Updated ML models for {condition}: RF={rf_score:.3f}, NN={nn_score:.3f}")
                        
                        # Store feature importance
                        if hasattr(rf_model, 'feature_importances_'):
                            self.feature_importance[condition] = rf_model.feature_importances_
            
        except Exception as e:
            logger.error(f"Error updating ML models: {e}")
    
    def _prepare_training_data(self, data_points: List) -> tuple:
        """Prepare training data from historical signals"""
        try:
            X = []
            y = []
            
            for point in data_points:
                signal = point.get('signal')
                if signal and hasattr(signal, 'market_condition'):
                    # Extract features (this would need the original analysis data)
                    # For now, using dummy features
                    features = np.random.rand(17)  # Match feature count
                    X.append(features)
                    
                    # Label based on action
                    if signal.action in ['sell_put', 'buy_call']:
                        y.append(1)  # Bullish
                    elif signal.action in ['sell_call', 'buy_put']:
                        y.append(0)  # Bearish
                    else:
                        y.append(0.5)  # Neutral
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for saving"""
        try:
            model_state = {
                'feature_importance': self.feature_importance,
                'model_performance': {}
            }
            
            for condition in self.condition_models:
                if hasattr(self.models[condition]['rf'], 'classes_'):
                    model_state['model_performance'][condition] = {
                        'rf_trained': True,
                        'nn_trained': hasattr(self.models[condition]['nn'], 'classes_')
                    }
            
            return model_state
            
        except Exception as e:
            logger.error(f"Error getting model state: {e}")
            return {}
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data.get('models', self.models)
            self.scalers = model_data.get('scalers', self.scalers)
            self.feature_importance = model_data.get('feature_importance', {})
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"Could not load models from {filepath}: {e}")
