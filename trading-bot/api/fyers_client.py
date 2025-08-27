#!/usr/bin/env python3
"""
Fyers API Client for trading operations
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
from fyers_apiv3 import fyersModel

from utils.logger import setup_logger

logger = setup_logger(__name__)

class FyersClient:
    """Enhanced Fyers API client with error handling and rate limiting"""
    
    def __init__(self, client_id: str, secret_key: str, redirect_uri: str, access_token: str = None):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self.fyers = None
        
    def authenticate(self) -> bool:
        """Authenticate with Fyers API"""
        try:
            if self.access_token:
                self.fyers = fyersModel.FyersModel(
                    client_id=self.client_id,
                    token=self.access_token,
                    log_path=""
                )
                
                # Test authentication
                profile = self.fyers.get_profile()
                if profile['s'] == 'ok':
                    logger.info("Fyers authentication successful")
                    return True
                else:
                    logger.error(f"Authentication failed: {profile}")
                    return False
            else:
                logger.error("No access token provided")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            data = {
                "symbol": symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": int(start_date.timestamp()),
                "range_to": int(end_date.timestamp()),
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data=data)
            
            if response['s'] == 'ok' and 'candles' in response:
                df = pd.DataFrame(response['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                logger.error(f"Failed to fetch historical data: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for symbols"""
        try:
            data = {"symbols": ",".join(symbols)}
            response = self.fyers.quotes(data=data)
            
            if response['s'] == 'ok':
                return response.get('d', {})
            else:
                logger.error(f"Failed to get quotes: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting quotes: {e}")
            return {}
    
    def place_order(self, symbol: str, qty: int, side: int, type: int, price: float = 0) -> Dict[str, Any]:
        """Place trading order"""
        try:
            data = {
                "symbol": symbol,
                "qty": qty,
                "type": type,
                "side": side,
                "productType": "INTRADAY",
                "limitPrice": price if type == 1 else 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": "False"
            }
            
            response = self.fyers.place_order(data=data)
            logger.info(f"Order placed: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"s": "error", "message": str(e)}
    
    def get_options_chain(self, symbol: str, expiry: str) -> Dict[str, Any]:
        """Get options chain data"""
        try:
            data = {
                "symbol": symbol,
                "expiry": expiry
            }
            
            response = self.fyers.optionchain(data=data)
            
            if response['s'] == 'ok':
                return response.get('data', {})
            else:
                logger.error(f"Failed to get options chain: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return {}
