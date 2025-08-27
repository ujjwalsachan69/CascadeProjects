#!/usr/bin/env python3
"""
Fyers API Client for Trading Bot
Handles authentication, data fetching, and order execution
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from fyers_apiv3 import fyersModel
import pytz

logger = logging.getLogger(__name__)

class FyersClient:
    """Fyers API client for data and trading operations"""
    
    def __init__(self, client_id: str, secret_key: str, redirect_uri: str, access_token: str = None):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self.fyers = None
        
        if access_token:
            self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
    
    def authenticate(self) -> bool:
        """Authenticate with Fyers API"""
        try:
            if not self.access_token:
                logger.error("Access token required for authentication")
                return False
            
            self.fyers = fyersModel.FyersModel(client_id=self.client_id, token=self.access_token)
            
            # Test authentication
            profile = self.fyers.get_profile()
            if profile['s'] == 'ok':
                logger.info(f"Successfully authenticated with Fyers API for user: {profile['data']['name']}")
                return True
            else:
                logger.error(f"Authentication failed: {profile}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data from Fyers API
        
        Args:
            symbol: Trading symbol (e.g., 'NSE:NIFTY50-INDEX')
            timeframe: '1' for 1-min, 'D' for daily
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        try:
            if not self.fyers:
                logger.error("Not authenticated with Fyers API")
                return pd.DataFrame()
            
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            data = {
                "symbol": symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": start_timestamp,
                "range_to": end_timestamp,
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data=data)
            
            if response['s'] == 'ok' and 'candles' in response:
                df = pd.DataFrame(response['candles'], 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
                return df
            else:
                logger.error(f"Failed to fetch data: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_options_chain(self, symbol: str, expiry_date: str = None) -> Dict:
        """
        Fetch options chain data
        
        Args:
            symbol: Underlying symbol (e.g., 'NIFTY')
            expiry_date: Expiry date in YYYY-MM-DD format
        """
        try:
            if not self.fyers:
                logger.error("Not authenticated with Fyers API")
                return {}
            
            data = {
                "symbol": f"NSE:{symbol}-INDEX",
                "strikecount": "10",
                "timestamp": expiry_date if expiry_date else ""
            }
            
            response = self.fyers.optionchain(data=data)
            
            if response['s'] == 'ok':
                return response['data']
            else:
                logger.error(f"Failed to fetch options chain: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return {}
    
    def get_quotes(self, symbols: List[str]) -> Dict:
        """Get real-time quotes for multiple symbols"""
        try:
            if not self.fyers:
                logger.error("Not authenticated with Fyers API")
                return {}
            
            data = {"symbols": ",".join(symbols)}
            response = self.fyers.quotes(data=data)
            
            if response['s'] == 'ok':
                return response['d']
            else:
                logger.error(f"Failed to fetch quotes: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            return {}
    
    def place_order(self, symbol: str, qty: int, side: int, type: int, price: float = 0) -> Dict:
        """
        Place an order
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            side: 1 for buy, -1 for sell
            type: 1 for limit, 2 for market, 3 for stop loss, 4 for stop loss market
            price: Price for limit orders
        """
        try:
            if not self.fyers:
                logger.error("Not authenticated with Fyers API")
                return {}
            
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
            
            if response['s'] == 'ok':
                logger.info(f"Order placed successfully: {response}")
                return response
            else:
                logger.error(f"Failed to place order: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if not self.fyers:
                logger.error("Not authenticated with Fyers API")
                return []
            
            response = self.fyers.positions()
            
            if response['s'] == 'ok':
                return response.get('netPositions', [])
            else:
                logger.error(f"Failed to fetch positions: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get order book"""
        try:
            if not self.fyers:
                logger.error("Not authenticated with Fyers API")
                return []
            
            response = self.fyers.orderbook()
            
            if response['s'] == 'ok':
                return response.get('orderBook', [])
            else:
                logger.error(f"Failed to fetch orders: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
