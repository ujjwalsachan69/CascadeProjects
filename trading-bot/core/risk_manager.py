#!/usr/bin/env python3
"""
=============================================================================
RISK MANAGER - Protects your money and controls position sizes
=============================================================================

PURPOSE: This file manages all risk-related aspects of trading:
- Calculates how much to trade per signal (position sizing)
- Sets stop losses and profit targets automatically
- Prevents trading when daily loss limits are reached
- Monitors total portfolio exposure and risk
- Tracks performance metrics like win rate and profit factor

KEY FUNCTIONS:
1. calculate_position_size() - Determines how many shares/contracts to trade
2. validate_trade() - Checks if trade meets risk criteria before execution
3. calculate_stop_loss() - Sets automatic stop loss levels
4. update_daily_pnl() - Tracks daily profit/loss
5. get_risk_status() - Shows current risk exposure

IMPORTANT SETTINGS YOU CAN EDIT:
=============================================================================
Line 45-55: RISK LIMITS - Maximum losses and position sizes
Line 60-70: STOP LOSS RULES - How tight or wide to set stop losses
Line 75-85: POSITION LIMITS - Maximum number of trades
Line 90-100: PROFIT TARGETS - When to take profits
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# RISK LIMITS - Line 45-55
# Edit these values to control your maximum risk exposure
# =============================================================================

@dataclass
class RiskParameters:
    """Risk management parameters - EDIT these values to control risk"""
    max_position_size: float = 100000      # EDIT: Maximum position size in rupees
    max_daily_loss: float = 5000           # EDIT: Maximum loss per day in rupees
    max_portfolio_risk: float = 0.02       # EDIT: Maximum portfolio risk (2% = 0.02)
    stop_loss_pct: float = 0.05            # EDIT: Stop loss percentage (5% = 0.05)
    profit_target_pct: float = 0.10        # EDIT: Profit target percentage (10% = 0.10)
    max_positions: int = 5                 # EDIT: Maximum number of open positions
    risk_per_trade: float = 0.01           # EDIT: Risk per trade (1% of portfolio = 0.01)

{{ ... }}
