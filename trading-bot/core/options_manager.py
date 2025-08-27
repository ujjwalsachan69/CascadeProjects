#!/usr/bin/env python3
"""
=============================================================================
OPTIONS MANAGER - Handles all options trading operations
=============================================================================

PURPOSE: This file manages everything related to options trading:
- Fetches live options chain data from Fyers API
- Selects the best options to buy/sell based on strategy
- Calculates option Greeks (Delta, Gamma, Theta, Vega)
- Finds support/resistance levels using options data
- Manages options expiry dates and strike selection

KEY FUNCTIONS:
1. fetch_options_chain() - Gets all available options for a symbol
2. select_options_for_selling() - Chooses best options to sell
3. calculate_option_greeks() - Computes Delta, Gamma, Theta, Vega
4. get_option_chain_analysis() - Analyzes Put-Call ratio and max pain
5. build_option_symbol() - Creates proper option symbols for trading

IMPORTANT SETTINGS YOU CAN EDIT:
=============================================================================
Line 65-75: STRIKE SELECTION - How far OTM to select options
Line 80-90: LIQUIDITY FILTERS - Minimum volume and open interest
Line 95-105: EXPIRY PREFERENCES - Which expiry dates to prefer
Line 110-120: GREEKS THRESHOLDS - Delta, Theta limits for selection
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from api.fyers_client import FyersClient
from utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# STRIKE SELECTION RULES - Line 65-75
# Edit these to change how far out-of-money to select options
# =============================================================================

STRIKE_SELECTION_CONFIG = {
    'otm_percentage_range': {              # EDIT: How far OTM to look for options
        'min_otm': 0.02,                   # Minimum 2% out-of-money
        'max_otm': 0.15,                   # Maximum 15% out-of-money
        'preferred_otm': 0.05              # Preferred 5% out-of-money
    },
    'strike_intervals': {                  # EDIT: Strike price intervals
        'NIFTY': 50,                       # Nifty strikes in multiples of 50
        'BANKNIFTY': 100,                  # Bank Nifty strikes in multiples of 100
        'FINNIFTY': 50                     # Fin Nifty strikes in multiples of 50
    },
    'atm_buffer': 0.005                    # EDIT: Buffer around ATM (0.5% = 0.005)
}

# =============================================================================
# LIQUIDITY FILTERS - Line 80-90
# Edit these to ensure you only trade liquid options
# =============================================================================

LIQUIDITY_CONFIG = {
    'min_volume': 100,                     # EDIT: Minimum daily volume
    'min_open_interest': 1000,             # EDIT: Minimum open interest
    'min_bid_ask_volume': 50,              # EDIT: Minimum bid/ask volume
    'max_bid_ask_spread': 0.05,            # EDIT: Maximum bid-ask spread (5%)
    'volume_percentile': 0.25,             # EDIT: Minimum volume percentile (top 75%)
    'oi_percentile': 0.30                  # EDIT: Minimum OI percentile (top 70%)
}

# =============================================================================
# EXPIRY PREFERENCES - Line 95-105
# Edit these to choose which expiry dates to prefer
# =============================================================================

EXPIRY_CONFIG = {
    'preferred_expiry_days': {             # EDIT: Preferred days to expiry
        'min_days': 1,                     # Minimum 1 day to expiry
        'max_days': 30,                    # Maximum 30 days to expiry
        'sweet_spot': 7                    # Sweet spot: 7 days to expiry
    },
    'expiry_day_mapping': {                # EDIT: Expiry days for different indices
        'NIFTY': 3,                        # Thursday (0=Monday)
        'BANKNIFTY': 2,                    # Wednesday
        'FINNIFTY': 1                      # Tuesday
    },
    'avoid_expiry_day': True,              # EDIT: Avoid trading on expiry day
    'early_exit_before_expiry': 2          # EDIT: Exit 2 hours before expiry
}

# =============================================================================
# GREEKS THRESHOLDS - Line 110-120
# Edit these to control option Greeks-based selection
# =============================================================================

GREEKS_CONFIG = {
    'delta_limits': {                      # EDIT: Delta limits for selection
        'max_short_delta': 0.30,           # Max delta when selling options
        'min_long_delta': 0.40,            # Min delta when buying options
        'delta_neutral_range': 0.10        # Delta neutral range (+/- 0.10)
    },
    'theta_preferences': {                 # EDIT: Theta (time decay) preferences
        'min_theta_decay': 0.01,           # Minimum daily theta decay
        'theta_acceleration_days': 7,       # Days when theta accelerates
        'max_theta_risk': 0.05             # Maximum theta risk per position
    },
    'vega_limits': {                       # EDIT: Vega (volatility) limits
        'max_vega_exposure': 0.25,         # Maximum vega per position
        'vega_hedge_threshold': 0.50,      # When to hedge vega exposure
        'iv_percentile_min': 0.30          # Minimum IV percentile to sell
    },
    'gamma_controls': {                    # EDIT: Gamma (acceleration) controls
        'max_gamma_risk': 0.15,            # Maximum gamma per position
        'gamma_scalping_threshold': 0.20,   # When to gamma scalp
        'avoid_high_gamma_days': 3         # Avoid high gamma X days before expiry
    }
}

@dataclass
class OptionContract:
    """Option contract data structure"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'CE' or 'PE'
    ltp: float
    volume: int
    oi: int
    bid: float
    ask: float
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

class OptionsManager:
    """Enhanced options manager with comprehensive options trading capabilities"""
    
    def __init__(self, fyers_client: FyersClient):
        self.fyers = fyers_client
        self.logger = setup_logger('options_manager')
        
        # Options symbols mapping
        self.symbols_map = {
            'NIFTY': 'NSE:NIFTY50-INDEX',
            'BANKNIFTY': 'NSE:NIFTYBANK-INDEX',
            'FINNIFTY': 'NSE:NIFTYFIN-INDEX'
        }
        
        # Options chain cache
        self.options_cache = {}
        self.cache_expiry = {}
        
    def get_next_expiry_dates(self, symbol: str, count: int = 3) -> List[str]:
        """Get next expiry dates for the symbol"""
        try:
            # For Nifty options, expiry is weekly on Thursdays
            # For Bank Nifty, expiry is weekly on Wednesdays
            # For Fin Nifty, expiry is weekly on Tuesdays
            
            today = datetime.now()
            expiry_dates = []
            
            # Determine expiry day based on symbol
            if symbol.upper() == 'NIFTY':
                expiry_day = 3  # Thursday (0=Monday)
            elif symbol.upper() == 'BANKNIFTY':
                expiry_day = 2  # Wednesday
            elif symbol.upper() == 'FINNIFTY':
                expiry_day = 1  # Tuesday
            else:
                expiry_day = 3  # Default to Thursday
            
            # Find next expiry dates
            current_date = today
            while len(expiry_dates) < count:
                days_ahead = expiry_day - current_date.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                
                expiry_date = current_date + timedelta(days=days_ahead)
                expiry_str = expiry_date.strftime('%Y-%m-%d')
                expiry_dates.append(expiry_str)
                
                # Move to next week
                current_date = expiry_date + timedelta(days=1)
            
            self.logger.info(f"Next {count} expiry dates for {symbol}: {expiry_dates}")
            return expiry_dates
            
        except Exception as e:
            self.logger.error(f"Error getting expiry dates for {symbol}: {e}")
            return []
    
    def fetch_options_chain(self, symbol: str, expiry: str) -> Dict[str, List[OptionContract]]:
        """Fetch complete options chain for symbol and expiry"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{expiry}"
            if (cache_key in self.options_cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                self.logger.info(f"Using cached options chain for {symbol} {expiry}")
                return self.options_cache[cache_key]
            
            # Get current price for ATM calculation
            current_price = self.get_current_price(symbol)
            if not current_price:
                self.logger.error(f"Could not get current price for {symbol}")
                return {'CE': [], 'PE': []}
            
            # Calculate strike range (ATM ± 20%)
            atm_strike = round(current_price / 50) * 50  # Round to nearest 50
            strike_range = int(current_price * 0.2)
            min_strike = atm_strike - strike_range
            max_strike = atm_strike + strike_range
            
            # Generate strike prices
            strikes = list(range(int(min_strike), int(max_strike) + 50, 50))
            
            options_chain = {'CE': [], 'PE': []}
            
            # Fetch options data for each strike
            for strike in strikes:
                # Call option
                ce_symbol = self.build_option_symbol(symbol, expiry, strike, 'CE')
                ce_data = self.get_option_quote(ce_symbol)
                if ce_data:
                    ce_contract = OptionContract(
                        symbol=ce_symbol,
                        strike=strike,
                        expiry=expiry,
                        option_type='CE',
                        ltp=ce_data.get('ltp', 0),
                        volume=ce_data.get('volume', 0),
                        oi=ce_data.get('oi', 0),
                        bid=ce_data.get('bid', 0),
                        ask=ce_data.get('ask', 0)
                    )
                    options_chain['CE'].append(ce_contract)
                
                # Put option
                pe_symbol = self.build_option_symbol(symbol, expiry, strike, 'PE')
                pe_data = self.get_option_quote(pe_symbol)
                if pe_data:
                    pe_contract = OptionContract(
                        symbol=pe_symbol,
                        strike=strike,
                        expiry=expiry,
                        option_type='PE',
                        ltp=pe_data.get('ltp', 0),
                        volume=pe_data.get('volume', 0),
                        oi=pe_data.get('oi', 0),
                        bid=pe_data.get('bid', 0),
                        ask=pe_data.get('ask', 0)
                    )
                    options_chain['PE'].append(pe_contract)
            
            # Cache the result for 5 minutes
            self.options_cache[cache_key] = options_chain
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
            
            self.logger.info(f"Fetched options chain for {symbol} {expiry}: "
                           f"{len(options_chain['CE'])} CEs, {len(options_chain['PE'])} PEs")
            
            return options_chain
            
        except Exception as e:
            self.logger.error(f"Error fetching options chain for {symbol} {expiry}: {e}")
            return {'CE': [], 'PE': []}
    
    def build_option_symbol(self, underlying: str, expiry: str, strike: float, option_type: str) -> str:
        """Build Fyers option symbol format"""
        try:
            # Convert expiry date format
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            expiry_str = expiry_date.strftime('%y%m%d')
            
            # Format strike price
            strike_str = f"{int(strike):05d}"
            
            # Build symbol based on underlying
            if underlying.upper() == 'NIFTY':
                symbol = f"NSE:NIFTY{expiry_str}{strike_str}{option_type}"
            elif underlying.upper() == 'BANKNIFTY':
                symbol = f"NSE:BANKNIFTY{expiry_str}{strike_str}{option_type}"
            elif underlying.upper() == 'FINNIFTY':
                symbol = f"NSE:FINNIFTY{expiry_str}{strike_str}{option_type}"
            else:
                symbol = f"NSE:{underlying.upper()}{expiry_str}{strike_str}{option_type}"
            
            return symbol
            
        except Exception as e:
            self.logger.error(f"Error building option symbol: {e}")
            return ""
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price of underlying"""
        try:
            if symbol.upper() in self.symbols_map:
                fyers_symbol = self.symbols_map[symbol.upper()]
                quote = self.fyers.get_quotes([fyers_symbol])
                
                if quote and 'd' in quote and len(quote['d']) > 0:
                    return quote['d'][0]['v']['lp']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for option"""
        try:
            quote = self.fyers.get_quotes([option_symbol])
            
            if quote and 'd' in quote and len(quote['d']) > 0:
                data = quote['d'][0]['v']
                return {
                    'ltp': data.get('lp', 0),
                    'volume': data.get('volume', 0),
                    'oi': data.get('oi', 0),
                    'bid': data.get('bp1', 0),
                    'ask': data.get('ap1', 0),
                    'change': data.get('ch', 0),
                    'change_pct': data.get('chp', 0)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting quote for {option_symbol}: {e}")
            return None
    
    def select_options_for_selling(self, symbol: str, current_price: float, 
                                 strategy: str = 'otm_put') -> List[OptionContract]:
        """Select options for selling based on strategy"""
        try:
            expiry_dates = self.get_next_expiry_dates(symbol, 1)
            if not expiry_dates:
                return []
            
            options_chain = self.fetch_options_chain(symbol, expiry_dates[0])
            
            selected_options = []
            
            if strategy == 'otm_put':
                # Select OTM puts for selling
                otm_puts = [opt for opt in options_chain['PE'] if opt.strike < current_price]
                # Filter by liquidity and premium
                liquid_puts = [opt for opt in otm_puts 
                             if opt.volume > 100 and opt.ltp > 10 and opt.oi > 1000]
                # Sort by premium and select top option
                liquid_puts.sort(key=lambda x: x.ltp, reverse=True)
                if liquid_puts:
                    selected_options.append(liquid_puts[0])
            
            elif strategy == 'otm_call':
                # Select OTM calls for selling
                otm_calls = [opt for opt in options_chain['CE'] if opt.strike > current_price]
                liquid_calls = [opt for opt in otm_calls 
                              if opt.volume > 100 and opt.ltp > 10 and opt.oi > 1000]
                liquid_calls.sort(key=lambda x: x.ltp, reverse=True)
                if liquid_calls:
                    selected_options.append(liquid_calls[0])
            
            elif strategy == 'strangle':
                # Select both OTM put and call
                otm_puts = [opt for opt in options_chain['PE'] 
                          if opt.strike < current_price and opt.volume > 100 and opt.ltp > 10]
                otm_calls = [opt for opt in options_chain['CE'] 
                           if opt.strike > current_price and opt.volume > 100 and opt.ltp > 10]
                
                if otm_puts and otm_calls:
                    # Select based on similar distance from ATM
                    atm_distance = 200  # Adjust based on volatility
                    put_strike = current_price - atm_distance
                    call_strike = current_price + atm_distance
                    
                    # Find closest strikes
                    best_put = min(otm_puts, key=lambda x: abs(x.strike - put_strike))
                    best_call = min(otm_calls, key=lambda x: abs(x.strike - call_strike))
                    
                    selected_options.extend([best_put, best_call])
            
            elif strategy == 'iron_condor':
                # Select 4 options for iron condor
                # This is more complex and requires careful strike selection
                pass
            
            self.logger.info(f"Selected {len(selected_options)} options for {strategy} strategy")
            return selected_options
            
        except Exception as e:
            self.logger.error(f"Error selecting options for {strategy}: {e}")
            return []
    
    def calculate_option_greeks(self, option: OptionContract, underlying_price: float, 
                              risk_free_rate: float = 0.06) -> OptionContract:
        """Calculate option Greeks (simplified Black-Scholes)"""
        try:
            # This is a simplified implementation
            # In production, use a proper options pricing library
            
            from math import log, sqrt, exp
            from scipy.stats import norm
            
            S = underlying_price  # Current price
            K = option.strike     # Strike price
            T = self.get_time_to_expiry(option.expiry)  # Time to expiry
            r = risk_free_rate    # Risk-free rate
            sigma = self.estimate_implied_volatility(option, S)  # Implied volatility
            
            if T <= 0 or sigma <= 0:
                return option
            
            # Calculate d1 and d2
            d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)
            
            if option.option_type == 'CE':
                # Call option
                option.delta = norm.cdf(d1)
                option.gamma = norm.pdf(d1) / (S*sigma*sqrt(T))
                option.theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)
                option.vega = S*norm.pdf(d1)*sqrt(T)
            else:
                # Put option
                option.delta = -norm.cdf(-d1)
                option.gamma = norm.pdf(d1) / (S*sigma*sqrt(T))
                option.theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2)
                option.vega = S*norm.pdf(d1)*sqrt(T)
            
            return option
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return option
    
    def get_time_to_expiry(self, expiry: str) -> float:
        """Calculate time to expiry in years"""
        try:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            now = datetime.now()
            time_diff = expiry_date - now
            return max(time_diff.total_seconds() / (365.25 * 24 * 3600), 0)
        except:
            return 0
    
    def estimate_implied_volatility(self, option: OptionContract, underlying_price: float) -> float:
        """Estimate implied volatility (simplified)"""
        try:
            # This is a very simplified estimation
            # In production, use Newton-Raphson or other numerical methods
            
            # Use historical volatility as proxy
            # You would typically fetch historical data and calculate
            
            # Default volatility estimates based on underlying
            if 'NIFTY' in option.symbol:
                return 0.20  # 20% volatility
            elif 'BANK' in option.symbol:
                return 0.25  # 25% volatility
            else:
                return 0.22  # 22% volatility
                
        except:
            return 0.20
    
    def get_option_chain_analysis(self, symbol: str, expiry: str) -> Dict[str, Any]:
        """Comprehensive options chain analysis"""
        try:
            options_chain = self.fetch_options_chain(symbol, expiry)
            current_price = self.get_current_price(symbol)
            
            if not current_price:
                return {}
            
            analysis = {
                'symbol': symbol,
                'expiry': expiry,
                'underlying_price': current_price,
                'total_ce_volume': sum(opt.volume for opt in options_chain['CE']),
                'total_pe_volume': sum(opt.volume for opt in options_chain['PE']),
                'total_ce_oi': sum(opt.oi for opt in options_chain['CE']),
                'total_pe_oi': sum(opt.oi for opt in options_chain['PE']),
                'max_pain': self.calculate_max_pain(options_chain),
                'pcr_volume': 0,
                'pcr_oi': 0,
                'support_levels': [],
                'resistance_levels': []
            }
            
            # Calculate Put-Call Ratios
            if analysis['total_ce_volume'] > 0:
                analysis['pcr_volume'] = analysis['total_pe_volume'] / analysis['total_ce_volume']
            
            if analysis['total_ce_oi'] > 0:
                analysis['pcr_oi'] = analysis['total_pe_oi'] / analysis['total_ce_oi']
            
            # Find support and resistance levels
            analysis['support_levels'] = self.find_support_levels(options_chain, current_price)
            analysis['resistance_levels'] = self.find_resistance_levels(options_chain, current_price)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in options chain analysis: {e}")
            return {}
    
    def calculate_max_pain(self, options_chain: Dict[str, List[OptionContract]]) -> float:
        """Calculate max pain point"""
        try:
            strikes = set()
            for opt in options_chain['CE'] + options_chain['PE']:
                strikes.add(opt.strike)
            
            max_pain_strike = 0
            min_pain = float('inf')
            
            for strike in strikes:
                total_pain = 0
                
                # Calculate pain for calls
                for ce in options_chain['CE']:
                    if strike > ce.strike:
                        total_pain += (strike - ce.strike) * ce.oi
                
                # Calculate pain for puts
                for pe in options_chain['PE']:
                    if strike < pe.strike:
                        total_pain += (pe.strike - strike) * pe.oi
                
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike
            
            return max_pain_strike
            
        except Exception as e:
            self.logger.error(f"Error calculating max pain: {e}")
            return 0
    
    def find_support_levels(self, options_chain: Dict[str, List[OptionContract]], 
                          current_price: float) -> List[float]:
        """Find support levels based on put OI"""
        try:
            put_oi = {}
            for pe in options_chain['PE']:
                if pe.strike < current_price:
                    put_oi[pe.strike] = pe.oi
            
            # Sort by OI and return top 3 strikes
            sorted_puts = sorted(put_oi.items(), key=lambda x: x[1], reverse=True)
            return [strike for strike, oi in sorted_puts[:3]]
            
        except Exception as e:
            self.logger.error(f"Error finding support levels: {e}")
            return []
    
    def find_resistance_levels(self, options_chain: Dict[str, List[OptionContract]], 
                             current_price: float) -> List[float]:
        """Find resistance levels based on call OI"""
        try:
            call_oi = {}
            for ce in options_chain['CE']:
                if ce.strike > current_price:
                    call_oi[ce.strike] = ce.oi
            
            # Sort by OI and return top 3 strikes
            sorted_calls = sorted(call_oi.items(), key=lambda x: x[1], reverse=True)
            return [strike for strike, oi in sorted_calls[:3]]
            
        except Exception as e:
            self.logger.error(f"Error finding resistance levels: {e}")
            return []
