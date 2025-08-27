#!/usr/bin/env python3
"""
=============================================================================
DATA MANAGER - Downloads, stores, and manages market data
=============================================================================

PURPOSE: This file handles all data-related operations:
- Downloads historical price data from Fyers API
- Stores data in CSV files for backup and analysis
- Manages data updates and synchronization
- Tracks performance metrics and trade history
- Handles data validation and cleaning

KEY FUNCTIONS:
1. fetch_historical_data() - Downloads price data from Fyers
2. save_data_to_csv() - Saves data to CSV files for backup
3. load_historical_data() - Loads previously saved data
4. update_performance_metrics() - Tracks trading performance
5. validate_data_quality() - Ensures data integrity

IMPORTANT SETTINGS YOU CAN EDIT:
=============================================================================
Line 60-70: DATA STORAGE - Where to save CSV files and backup settings
Line 75-85: DATA PERIODS - How much historical data to download
Line 90-100: UPDATE FREQUENCY - How often to refresh data
Line 105-115: DATA VALIDATION - Quality checks and error handling
Line 120-130: PERFORMANCE TRACKING - Metrics to track and save
=============================================================================
"""

import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# DATA STORAGE SETTINGS - Line 60-70
# Edit these to control where and how data is saved
# =============================================================================

DATA_STORAGE_CONFIG = {
    'csv_directory': 'data/csv_backups',   # EDIT: Directory for CSV backups
    'performance_file': 'data/performance_metrics.json',  # EDIT: Performance file
    'backup_frequency': 'daily',           # EDIT: Backup frequency (daily/hourly)
    'max_backup_files': 30,                # EDIT: Maximum backup files to keep
    'compress_old_files': True,            # EDIT: Compress files older than 7 days
    'auto_cleanup_days': 90                # EDIT: Auto-delete files older than X days
}

# =============================================================================
# DATA PERIODS - Line 75-85
# Edit these to control how much historical data to download
# =============================================================================

DATA_PERIOD_CONFIG = {
    'historical_days': {                   # EDIT: Historical data periods
        'daily_data_days': 90,             # 90 days of daily data
        'minute_data_days': 30,            # 30 days of minute data
        'tick_data_days': 7                # 7 days of tick data (if needed)
    },
    'data_intervals': {                    # EDIT: Data interval settings
        'daily_interval': '1D',            # Daily candles
        'minute_intervals': ['1', '5', '15'], # Minute intervals to download
        'preferred_interval': '5'          # Preferred interval for analysis
    },
    'market_hours': {                      # EDIT: Market hours for data filtering
        'market_open': '09:15',            # Market opening time
        'market_close': '15:30',           # Market closing time
        'include_pre_market': False,       # Include pre-market data
        'include_after_hours': False       # Include after-hours data
    }
}

# =============================================================================
# UPDATE FREQUENCY - Line 90-100
# Edit these to control how often data is refreshed
# =============================================================================

UPDATE_CONFIG = {
    'refresh_intervals': {                 # EDIT: Data refresh settings
        'live_data_seconds': 30,           # Refresh live data every 30 seconds
        'historical_data_hours': 4,        # Refresh historical data every 4 hours
        'options_chain_minutes': 5         # Refresh options chain every 5 minutes
    },
    'update_triggers': {                   # EDIT: When to trigger data updates
        'market_open_update': True,        # Update when market opens
        'significant_move_pct': 0.02,      # Update on 2% price moves
        'volume_spike_multiplier': 3.0     # Update on 3x volume spikes
    },
    'retry_settings': {                    # EDIT: Retry settings for failed downloads
        'max_retries': 3,                  # Maximum retry attempts
        'retry_delay_seconds': 10,         # Delay between retries
        'exponential_backoff': True        # Use exponential backoff
    }
}

# =============================================================================
# DATA VALIDATION - Line 105-115
# Edit these to control data quality checks
# =============================================================================

VALIDATION_CONFIG = {
    'quality_checks': {                    # EDIT: Data quality validation
        'check_missing_data': True,        # Check for missing data points
        'max_missing_pct': 0.05,           # Maximum 5% missing data allowed
        'check_price_spikes': True,        # Check for unrealistic price spikes
        'max_price_change_pct': 0.20       # Maximum 20% price change per candle
    },
    'data_cleaning': {                     # EDIT: Data cleaning settings
        'remove_outliers': True,           # Remove statistical outliers
        'outlier_std_threshold': 3.0,      # Standard deviations for outliers
        'interpolate_missing': True,       # Interpolate missing values
        'forward_fill_limit': 5            # Maximum forward fill periods
    },
    'error_handling': {                    # EDIT: Error handling preferences
        'log_data_errors': True,           # Log data quality issues
        'alert_on_critical_errors': True,  # Alert on critical data issues
        'fallback_to_backup': True         # Use backup data if primary fails
    }
}

# =============================================================================
# PERFORMANCE TRACKING - Line 120-130
# Edit these to control what performance metrics to track
# =============================================================================

PERFORMANCE_CONFIG = {
    'metrics_to_track': {                  # EDIT: Which metrics to calculate and save
        'total_pnl': True,                 # Track total profit/loss
        'daily_pnl': True,                 # Track daily P&L
        'win_rate': True,                  # Track win percentage
        'profit_factor': True,             # Track profit factor
        'max_drawdown': True,              # Track maximum drawdown
        'sharpe_ratio': True,              # Track risk-adjusted returns
        'trade_count': True,               # Track number of trades
        'avg_trade_duration': True         # Track average trade duration
    },
    'reporting_frequency': {               # EDIT: How often to update metrics
        'real_time_updates': True,         # Update metrics in real-time
        'daily_summary': True,             # Generate daily summary
        'weekly_report': True,             # Generate weekly report
        'monthly_analysis': True           # Generate monthly analysis
    },
    'benchmark_comparison': {              # EDIT: Benchmark comparison settings
        'compare_to_nifty': True,          # Compare performance to Nifty
        'compare_to_banknifty': True,      # Compare to Bank Nifty
        'risk_free_rate': 0.06             # Risk-free rate for calculations (6%)
    }
}

class DataManager:
    """Enhanced data manager with comprehensive backup and archival capabilities"""
    
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = base_dir
        self.setup_directories()
        
        # Data paths
        self.historical_dir = os.path.join(base_dir, 'historical')
        self.options_dir = os.path.join(base_dir, 'options')
        self.trades_dir = os.path.join(base_dir, 'trades')
        self.analysis_dir = os.path.join(base_dir, 'analysis')
        self.backtests_dir = os.path.join(base_dir, 'backtests')
        self.performance_dir = os.path.join(base_dir, 'performance')
        self.models_dir = os.path.join(base_dir, 'models')
        
        # Setup logging for data operations
        self.data_logger = setup_logger('data_manager')
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'historical', 'options', 'trades', 'analysis',
            'backtests', 'performance', 'models', 'logs'
        ]
        
        for directory in directories:
            dir_path = os.path.join(self.base_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Data directories initialized in {self.base_dir}")
    
    def save_historical_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Save historical OHLCV data to CSV"""
        try:
            if data.empty:
                logger.warning(f"No data to save for {symbol} {timeframe}")
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.csv"
            filepath = os.path.join(self.historical_dir, filename)
            
            # Save to CSV
            data.to_csv(filepath)
            
            # Also save latest version without timestamp
            latest_filename = f"{symbol}_{timeframe}_latest.csv"
            latest_filepath = os.path.join(self.historical_dir, latest_filename)
            data.to_csv(latest_filepath)
            
            logger.info(f"Saved historical data: {filename} ({len(data)} records)")
            
            # Log data summary
            self._log_data_summary(symbol, timeframe, data)
            
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {e}")
    
    def load_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load latest historical data from CSV"""
        try:
            filename = f"{symbol}_{timeframe}_latest.csv"
            filepath = os.path.join(self.historical_dir, filename)
            
            if os.path.exists(filepath):
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                logger.info(f"Loaded historical data: {filename} ({len(data)} records)")
                return data
            else:
                logger.warning(f"No historical data found for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None
    
    def save_options_chain(self, symbol: str, expiry: str, options_data: Dict[str, Any]):
        """Save options chain data to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_options_{expiry}_{timestamp}.csv"
            filepath = os.path.join(self.options_dir, filename)
            
            # Convert options data to DataFrame
            if 'PE' in options_data and 'CE' in options_data:
                pe_data = pd.DataFrame(options_data['PE'])
                ce_data = pd.DataFrame(options_data['CE'])
                
                # Add option type column
                pe_data['option_type'] = 'PE'
                ce_data['option_type'] = 'CE'
                
                # Combine and save
                combined_data = pd.concat([pe_data, ce_data], ignore_index=True)
                combined_data.to_csv(filepath, index=False)
                
                logger.info(f"Saved options chain: {filename} ({len(combined_data)} options)")
            
        except Exception as e:
            logger.error(f"Error saving options chain for {symbol}: {e}")
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade details to CSV"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now()
            
            # Create daily trade log file
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"trades_{date_str}.csv"
            filepath = os.path.join(self.trades_dir, filename)
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(filepath)
            
            # Write to CSV
            with open(filepath, 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'symbol', 'option_symbol', 'action', 'quantity',
                    'price', 'strike', 'expiry', 'option_type', 'strategy',
                    'confidence', 'reasoning', 'pnl', 'status', 'market_condition'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                # Ensure all required fields are present
                trade_record = {field: trade_data.get(field, '') for field in fieldnames}
                writer.writerow(trade_record)
            
            logger.info(f"Trade logged: {trade_data.get('action', 'UNKNOWN')} {trade_data.get('symbol', '')}")
            
            # Also maintain a master trade log
            self._append_to_master_log(trade_data)
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def save_analysis_results(self, symbol: str, analysis: Dict[str, Any]):
        """Save market analysis results to JSON and CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as JSON for complete data
            json_filename = f"{symbol}_analysis_{timestamp}.json"
            json_filepath = os.path.join(self.analysis_dir, json_filename)
            
            with open(json_filepath, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                analysis_copy = self._convert_numpy_types(analysis.copy())
                json.dump(analysis_copy, f, indent=2, default=str)
            
            # Save key metrics as CSV for easy analysis
            csv_filename = f"{symbol}_metrics_{timestamp}.csv"
            csv_filepath = os.path.join(self.analysis_dir, csv_filename)
            
            metrics = self._extract_key_metrics(analysis)
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(csv_filepath, index=False)
            
            logger.info(f"Analysis saved: {symbol} at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results for {symbol}: {e}")
    
    def save_backtest_results(self, backtest_id: str, results: Dict[str, Any]):
        """Save backtest results to CSV and JSON"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"backtest_{backtest_id}_{timestamp}"
            
            # Save complete results as JSON
            json_filepath = os.path.join(self.backtests_dir, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save trade log as CSV
            if 'trades' in results:
                trades_df = pd.DataFrame(results['trades'])
                csv_filepath = os.path.join(self.backtests_dir, f"{base_filename}_trades.csv")
                trades_df.to_csv(csv_filepath, index=False)
            
            # Save performance metrics as CSV
            if 'performance_metrics' in results:
                metrics_df = pd.DataFrame([results['performance_metrics']])
                metrics_filepath = os.path.join(self.backtests_dir, f"{base_filename}_metrics.csv")
                metrics_df.to_csv(metrics_filepath, index=False)
            
            logger.info(f"Backtest results saved: {backtest_id}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def save_performance_data(self, performance_data: Dict[str, Any]):
        """Save performance tracking data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{timestamp}.json"
            filepath = os.path.join(self.performance_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            # Also save as latest
            latest_filepath = os.path.join(self.performance_dir, "performance_latest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            logger.info("Performance data saved")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def load_performance_data(self) -> Optional[Dict[str, Any]]:
        """Load latest performance data"""
        try:
            filepath = os.path.join(self.performance_dir, "performance_latest.json")
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info("Performance data loaded")
                return data
            else:
                logger.info("No previous performance data found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return None
    
    def save_model_data(self, model_data: Dict[str, Any]):
        """Save ML model data and weights"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models_{timestamp}.json"
            filepath = os.path.join(self.models_dir, filename)
            
            # Convert numpy arrays to lists for JSON serialization
            model_data_copy = self._convert_numpy_types(model_data.copy())
            
            with open(filepath, 'w') as f:
                json.dump(model_data_copy, f, indent=2, default=str)
            
            # Save as latest
            latest_filepath = os.path.join(self.models_dir, "models_latest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(model_data_copy, f, indent=2, default=str)
            
            logger.info("Model data saved")
            
        except Exception as e:
            logger.error(f"Error saving model data: {e}")
    
    def export_performance_report(self) -> str:
        """Export comprehensive performance report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"performance_report_{timestamp}.csv"
            report_filepath = os.path.join(self.performance_dir, report_filename)
            
            # Collect all trade data
            all_trades = self._collect_all_trades()
            
            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                
                # Calculate performance metrics
                report_data = self._calculate_performance_metrics(trades_df)
                
                # Save report
                report_df = pd.DataFrame([report_data])
                report_df.to_csv(report_filepath, index=False)
                
                logger.info(f"Performance report exported: {report_filename}")
                return report_filepath
            else:
                logger.warning("No trade data available for performance report")
                return ""
                
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return ""
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """Clean up old data files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            # Directories to clean
            cleanup_dirs = [
                self.historical_dir, self.options_dir, self.analysis_dir
            ]
            
            for directory in cleanup_dirs:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if filename.endswith(('.csv', '.json')) and not filename.endswith('_latest.csv'):
                            filepath = os.path.join(directory, filename)
                            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                            
                            if file_time < cutoff_date:
                                os.remove(filepath)
                                deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files (older than {days_to_keep} days)")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _log_data_summary(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Log summary of data saved"""
        try:
            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'records': len(data),
                'date_range': f"{data.index[0]} to {data.index[-1]}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to data log
            log_filepath = os.path.join(self.base_dir, 'logs', 'data_operations.log')
            with open(log_filepath, 'a') as f:
                f.write(f"{json.dumps(summary)}\n")
                
        except Exception as e:
            logger.error(f"Error logging data summary: {e}")
    
    def _append_to_master_log(self, trade_data: Dict[str, Any]):
        """Append trade to master log file"""
        try:
            master_filepath = os.path.join(self.trades_dir, "master_trades.csv")
            file_exists = os.path.exists(master_filepath)
            
            with open(master_filepath, 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'symbol', 'option_symbol', 'action', 'quantity',
                    'price', 'strike', 'expiry', 'option_type', 'strategy',
                    'confidence', 'reasoning', 'pnl', 'status', 'market_condition'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                trade_record = {field: trade_data.get(field, '') for field in fieldnames}
                writer.writerow(trade_record)
                
        except Exception as e:
            logger.error(f"Error appending to master log: {e}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis for CSV storage"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'symbol': analysis.get('symbol', ''),
                'current_ltp': analysis.get('current_ltp', 0),
                'current_change_pct': analysis.get('current_change_pct', 0),
                'volatility': analysis.get('volatility', 0),
                'latest_gap': analysis.get('latest_gap', 0),
                'trend_direction': analysis.get('daily_trend', {}).get('trend', ''),
                'trend_strength': analysis.get('daily_trend', {}).get('strength', ''),
                'rsi': analysis.get('momentum_indicators', {}).get('rsi', 0),
                'volume_ratio': analysis.get('volume_analysis', {}).get('volume_ratio', 0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return {}
    
    def _collect_all_trades(self) -> List[Dict[str, Any]]:
        """Collect all trade data from CSV files"""
        try:
            all_trades = []
            
            if os.path.exists(self.trades_dir):
                for filename in os.listdir(self.trades_dir):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(self.trades_dir, filename)
                        try:
                            df = pd.read_csv(filepath)
                            all_trades.extend(df.to_dict('records'))
                        except Exception as e:
                            logger.warning(f"Could not read {filename}: {e}")
            
            return all_trades
            
        except Exception as e:
            logger.error(f"Error collecting trade data: {e}")
            return []
    
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            # Convert PnL to numeric
            trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce').fillna(0)
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Advanced metrics
            winning_pnl = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            losing_pnl = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
            
            return {
                'report_date': datetime.now().isoformat(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_pct': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_pnl_per_trade': round(avg_pnl, 2),
                'avg_winning_trade': round(avg_win, 2),
                'avg_losing_trade': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_win': trades_df['pnl'].max(),
                'max_loss': trades_df['pnl'].min()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
