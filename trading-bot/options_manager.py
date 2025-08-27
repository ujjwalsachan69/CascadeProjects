#!/usr/bin/env python3
"""
Options Manager for Trading Bot
Handles options chain fetching, analysis, and selection
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Represents an option contract"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'CE' or 'PE'
    ltp: float
    bid: float
    ask: float
    volume: int
    oi: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float

class OptionsManager:
    """Manages options chain data and selection logic"""
    
    def __init__(self, fyers_client):
        self.fyers_client = fyers_client
        self.options_cache = {}
        self.cache_expiry = {}
    
    def get_next_expiry_dates(self, symbol: str, count: int = 3) -> List[datetime]:
        """Get next expiry dates for the given symbol"""
        try:
            # For Nifty and Bank Nifty, expiries are typically on Thursdays
            today = datetime.now()
            expiry_dates = []
            
            # Find next Thursday
            days_ahead = 3 - today.weekday()  # Thursday is 3
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_thursday = today + timedelta(days=days_ahead)
            
            # Get next few expiries
            for i in range(count):
                expiry_date = next_thursday + timedelta(weeks=i)
                expiry_dates.append(expiry_date)
            
            return expiry_dates
            
        except Exception as e:
            logger.error(f"Error getting expiry dates: {e}")
            return []
    
    def fetch_options_chain(self, symbol: str, expiry_date: datetime = None) -> Dict[str, List[OptionContract]]:
        """
        Fetch options chain for given symbol and expiry
        
        Returns:
            Dict with 'CE' and 'PE' keys containing lists of OptionContract objects
        """
        try:
            if expiry_date is None:
                expiry_dates = self.get_next_expiry_dates(symbol, 1)
                if not expiry_dates:
                    return {'CE': [], 'PE': []}
                expiry_date = expiry_dates[0]
            
            # Check cache
            cache_key = f"{symbol}_{expiry_date.strftime('%Y%m%d')}"
            if (cache_key in self.options_cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                return self.options_cache[cache_key]
            
            # Fetch from API
            expiry_str = expiry_date.strftime('%Y-%m-%d')
            options_data = self.fyers_client.get_options_chain(symbol, expiry_str)
            
            if not options_data:
                logger.warning(f"No options data received for {symbol}")
                return {'CE': [], 'PE': []}
            
            ce_contracts = []
            pe_contracts = []
            
            # Parse options data
            if 'optionsChain' in options_data:
                for strike_data in options_data['optionsChain']:
                    strike = float(strike_data.get('strikePrice', 0))
                    
                    # Call options (CE)
                    if 'call' in strike_data:
                        call_data = strike_data['call']
                        ce_contract = OptionContract(
                            symbol=call_data.get('symbol', ''),
                            strike=strike,
                            expiry=expiry_date,
                            option_type='CE',
                            ltp=float(call_data.get('ltp', 0)),
                            bid=float(call_data.get('bid', 0)),
                            ask=float(call_data.get('ask', 0)),
                            volume=int(call_data.get('volume', 0)),
                            oi=int(call_data.get('oi', 0)),
                            iv=float(call_data.get('iv', 0)),
                            delta=float(call_data.get('delta', 0)),
                            gamma=float(call_data.get('gamma', 0)),
                            theta=float(call_data.get('theta', 0)),
                            vega=float(call_data.get('vega', 0))
                        )
                        ce_contracts.append(ce_contract)
                    
                    # Put options (PE)
                    if 'put' in strike_data:
                        put_data = strike_data['put']
                        pe_contract = OptionContract(
                            symbol=put_data.get('symbol', ''),
                            strike=strike,
                            expiry=expiry_date,
                            option_type='PE',
                            ltp=float(put_data.get('ltp', 0)),
                            bid=float(put_data.get('bid', 0)),
                            ask=float(put_data.get('ask', 0)),
                            volume=int(put_data.get('volume', 0)),
                            oi=int(put_data.get('oi', 0)),
                            iv=float(put_data.get('iv', 0)),
                            delta=float(put_data.get('delta', 0)),
                            gamma=float(put_data.get('gamma', 0)),
                            theta=float(put_data.get('theta', 0)),
                            vega=float(put_data.get('vega', 0))
                        )
                        pe_contracts.append(pe_contract)
            
            options_chain = {'CE': ce_contracts, 'PE': pe_contracts}
            
            # Cache the data for 5 minutes
            self.options_cache[cache_key] = options_chain
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
            
            logger.info(f"Fetched options chain for {symbol}: {len(ce_contracts)} CE, {len(pe_contracts)} PE")
            return options_chain
            
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return {'CE': [], 'PE': []}
    
    def get_atm_strike(self, symbol: str, current_price: float) -> float:
        """Get the at-the-money strike price"""
        try:
            # Round to nearest strike (typically 50 point intervals for Nifty, 100 for Bank Nifty)
            if 'NIFTY' in symbol.upper() and 'BANK' not in symbol.upper():
                strike_interval = 50
            else:  # Bank Nifty
                strike_interval = 100
            
            atm_strike = round(current_price / strike_interval) * strike_interval
            return atm_strike
            
        except Exception as e:
            logger.error(f"Error calculating ATM strike: {e}")
            return current_price
    
    def select_options_for_selling(self, symbol: str, current_price: float, 
                                 strategy: str = 'otm_put') -> List[OptionContract]:
        """
        Select options for selling based on strategy
        
        Args:
            symbol: Underlying symbol (NIFTY or BANKNIFTY)
            current_price: Current price of underlying
            strategy: 'otm_put', 'otm_call', 'iron_condor', 'strangle'
        """
        try:
            # Get next expiry options
            expiry_dates = self.get_next_expiry_dates(symbol, 1)
            if not expiry_dates:
                return []
            
            options_chain = self.fetch_options_chain(symbol, expiry_dates[0])
            
            if not options_chain['CE'] and not options_chain['PE']:
                return []
            
            atm_strike = self.get_atm_strike(symbol, current_price)
            selected_options = []
            
            if strategy == 'otm_put':
                # Select OTM put for selling (below current price)
                pe_contracts = options_chain['PE']
                otm_puts = [opt for opt in pe_contracts if opt.strike < current_price]
                
                if otm_puts:
                    # Sort by strike descending and select highest strike OTM put
                    otm_puts.sort(key=lambda x: x.strike, reverse=True)
                    
                    # Filter by liquidity and premium
                    liquid_puts = [opt for opt in otm_puts[:5] if opt.volume > 100 and opt.ltp > 10]
                    
                    if liquid_puts:
                        selected_options.append(liquid_puts[0])
            
            elif strategy == 'otm_call':
                # Select OTM call for selling (above current price)
                ce_contracts = options_chain['CE']
                otm_calls = [opt for opt in ce_contracts if opt.strike > current_price]
                
                if otm_calls:
                    # Sort by strike ascending and select lowest strike OTM call
                    otm_calls.sort(key=lambda x: x.strike)
                    
                    # Filter by liquidity and premium
                    liquid_calls = [opt for opt in otm_calls[:5] if opt.volume > 100 and opt.ltp > 10]
                    
                    if liquid_calls:
                        selected_options.append(liquid_calls[0])
            
            elif strategy == 'strangle':
                # Sell both OTM call and OTM put
                otm_put_options = self.select_options_for_selling(symbol, current_price, 'otm_put')
                otm_call_options = self.select_options_for_selling(symbol, current_price, 'otm_call')
                selected_options.extend(otm_put_options)
                selected_options.extend(otm_call_options)
            
            elif strategy == 'iron_condor':
                # More complex strategy - sell closer strikes, buy farther strikes
                pe_contracts = options_chain['PE']
                ce_contracts = options_chain['CE']
                
                # Sell OTM put
                otm_puts = [opt for opt in pe_contracts if opt.strike < current_price]
                if otm_puts:
                    otm_puts.sort(key=lambda x: x.strike, reverse=True)
                    if otm_puts and otm_puts[0].volume > 50:
                        selected_options.append(otm_puts[0])
                
                # Sell OTM call
                otm_calls = [opt for opt in ce_contracts if opt.strike > current_price]
                if otm_calls:
                    otm_calls.sort(key=lambda x: x.strike)
                    if otm_calls and otm_calls[0].volume > 50:
                        selected_options.append(otm_calls[0])
            
            logger.info(f"Selected {len(selected_options)} options for {strategy} strategy")
            return selected_options
            
        except Exception as e:
            logger.error(f"Error selecting options: {e}")
            return []
    
    def calculate_option_metrics(self, option: OptionContract, underlying_price: float) -> Dict:
        """Calculate additional metrics for option analysis"""
        try:
            metrics = {}
            
            # Moneyness
            if option.option_type == 'CE':
                metrics['moneyness'] = underlying_price / option.strike
                metrics['intrinsic_value'] = max(0, underlying_price - option.strike)
            else:  # PE
                metrics['moneyness'] = option.strike / underlying_price
                metrics['intrinsic_value'] = max(0, option.strike - underlying_price)
            
            # Time value
            metrics['time_value'] = option.ltp - metrics['intrinsic_value']
            
            # Liquidity score (based on volume and open interest)
            metrics['liquidity_score'] = (option.volume * 0.3 + option.oi * 0.7) / 1000
            
            # Risk-reward ratio for selling
            days_to_expiry = (option.expiry - datetime.now()).days
            if days_to_expiry > 0:
                metrics['daily_theta'] = option.theta
                metrics['theta_efficiency'] = abs(option.theta) / option.ltp if option.ltp > 0 else 0
            else:
                metrics['daily_theta'] = 0
                metrics['theta_efficiency'] = 0
            
            # Probability of profit (rough estimate for selling)
            if option.option_type == 'CE':
                # For selling calls, profit if price stays below strike + premium
                breakeven = option.strike + option.ltp
                distance_pct = (breakeven - underlying_price) / underlying_price * 100
            else:
                # For selling puts, profit if price stays above strike - premium
                breakeven = option.strike - option.ltp
                distance_pct = (underlying_price - breakeven) / underlying_price * 100
            
            metrics['breakeven_distance_pct'] = distance_pct
            metrics['estimated_pop'] = min(90, max(10, 50 + distance_pct * 2))  # Rough estimate
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating option metrics: {e}")
            return {}
    
    def get_best_options_to_sell(self, symbol: str, current_price: float, 
                               max_options: int = 3) -> List[Tuple[OptionContract, Dict]]:
        """
        Get the best options to sell based on multiple criteria
        
        Returns:
            List of tuples (OptionContract, metrics_dict) sorted by attractiveness
        """
        try:
            # Get options for different strategies
            otm_puts = self.select_options_for_selling(symbol, current_price, 'otm_put')
            otm_calls = self.select_options_for_selling(symbol, current_price, 'otm_call')
            
            all_options = otm_puts + otm_calls
            
            if not all_options:
                return []
            
            # Calculate metrics for each option
            options_with_metrics = []
            for option in all_options:
                metrics = self.calculate_option_metrics(option, current_price)
                if metrics:
                    options_with_metrics.append((option, metrics))
            
            # Score and rank options
            for option, metrics in options_with_metrics:
                score = 0
                
                # Higher premium is better
                score += min(50, option.ltp * 2)
                
                # Higher theta efficiency is better
                score += metrics.get('theta_efficiency', 0) * 100
                
                # Higher liquidity is better
                score += min(20, metrics.get('liquidity_score', 0))
                
                # Higher probability of profit is better
                score += metrics.get('estimated_pop', 0) * 0.5
                
                # Prefer options with reasonable time value
                if metrics.get('time_value', 0) > 5:
                    score += 10
                
                metrics['total_score'] = score
            
            # Sort by score descending
            options_with_metrics.sort(key=lambda x: x[1]['total_score'], reverse=True)
            
            # Return top options
            return options_with_metrics[:max_options]
            
        except Exception as e:
            logger.error(f"Error getting best options to sell: {e}")
            return []
