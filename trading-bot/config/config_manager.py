#!/usr/bin/env python3
"""
Configuration Manager for centralized configuration handling
"""

import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

from utils.logger import setup_logger

logger = setup_logger(__name__)

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = 'config/config.json'):
        self.config_file = config_file
        self.config = {}
        load_dotenv()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
            
            # Override with environment variables
            self._load_from_env()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.config = config
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _load_from_env(self):
        """Load sensitive data from environment variables"""
        # Fyers API credentials
        if not self.config.get('fyers'):
            self.config['fyers'] = {}
        
        self.config['fyers']['client_id'] = os.getenv('FYERS_CLIENT_ID', 
                                                      self.config['fyers'].get('client_id', ''))
        self.config['fyers']['secret_key'] = os.getenv('FYERS_SECRET_KEY', 
                                                       self.config['fyers'].get('secret_key', ''))
        self.config['fyers']['redirect_uri'] = os.getenv('FYERS_REDIRECT_URI', 
                                                         self.config['fyers'].get('redirect_uri', 
                                                         'https://trade.fyers.in/api-login/redirect-to-app'))
        self.config['fyers']['access_token'] = os.getenv('FYERS_ACCESS_TOKEN', 
                                                         self.config['fyers'].get('access_token', ''))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'fyers': {
                'client_id': '',
                'secret_key': '',
                'redirect_uri': 'https://trade.fyers.in/api-login/redirect-to-app',
                'access_token': ''
            },
            'risk_management': {
                'max_position_value': 50000,
                'max_daily_loss': 10000,
                'max_positions': 3,
                'stop_loss_pct': 200,
                'profit_target_pct': 50,
                'portfolio_value': 100000
            },
            'trading': {
                'symbols': ['NIFTY', 'BANKNIFTY'],
                'cycle_interval': 5,
                'market_hours': {
                    'start': '09:15',
                    'end': '15:30'
                }
            },
            'ml_model': {
                'retrain_interval': 24,
                'min_data_points': 100,
                'model_types': ['random_forest', 'neural_network']
            },
            'logging': {
                'level': 'INFO',
                'max_file_size': 10485760,
                'backup_count': 5
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_fields = [
            'fyers.client_id',
            'fyers.secret_key',
            'fyers.access_token'
        ]
        
        for field in required_fields:
            if not self.get(field):
                logger.error(f"Missing required configuration: {field}")
                return False
        
        return True
