#!/usr/bin/env python3
"""
Data Manager for Trading Bot
Handles CSV backups, trade logs, and data archival
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data storage, backup, and logging for the trading bot"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
    
    def setup_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            self.base_dir,
            self.base_dir / "historical",
            self.base_dir / "trades",
            self.base_dir / "backtest",
            self.base_dir / "logs",
            self.base_dir / "analysis",
            self.base_dir / "options_chains"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def setup_logging(self):
        """Setup enhanced logging system with multiple log files"""
        log_dir = self.base_dir / "logs"
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Main trading log
        trading_handler = logging.FileHandler(log_dir / "trading.log")
        trading_handler.setFormatter(detailed_formatter)
        trading_handler.setLevel(logging.INFO)
        
        # Error log
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Data operations log
        data_handler = logging.FileHandler(log_dir / "data_operations.log")
        data_handler.setFormatter(simple_formatter)
        data_handler.setLevel(logging.INFO)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(trading_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(data_handler)
    
    def save_historical_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> str:
        """
        Save historical data to CSV with timestamp
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY')
            timeframe: Data timeframe ('daily', '1min', etc.)
            data: DataFrame with OHLCV data
        
        Returns:
            Path to saved file
        """
        try:
            if data.empty:
                logger.warning(f"No data to save for {symbol} {timeframe}")
                return ""
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.csv"
            filepath = self.base_dir / "historical" / filename
            
            # Add metadata columns
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            data_copy['timeframe'] = timeframe
            data_copy['saved_at'] = datetime.now()
            
            # Save to CSV
            data_copy.to_csv(filepath, index=True)
            
            logger.info(f"Saved historical data: {filepath} ({len(data)} records)")
            
            # Also save latest data (overwrite)
            latest_filepath = self.base_dir / "historical" / f"{symbol}_{timeframe}_latest.csv"
            data_copy.to_csv(latest_filepath, index=True)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {e}")
            return ""
    
    def save_options_chain(self, symbol: str, expiry: datetime, options_data: Dict) -> str:
        """Save options chain data to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            expiry_str = expiry.strftime("%Y%m%d")
            filename = f"{symbol}_options_{expiry_str}_{timestamp}.csv"
            filepath = self.base_dir / "options_chains" / filename
            
            # Convert options data to DataFrame
            all_options = []
            
            for option_type in ['CE', 'PE']:
                if option_type in options_data:
                    for option in options_data[option_type]:
                        option_dict = {
                            'symbol': option.symbol,
                            'underlying': symbol,
                            'strike': option.strike,
                            'expiry': option.expiry,
                            'option_type': option.option_type,
                            'ltp': option.ltp,
                            'bid': option.bid,
                            'ask': option.ask,
                            'volume': option.volume,
                            'oi': option.oi,
                            'iv': option.iv,
                            'delta': option.delta,
                            'gamma': option.gamma,
                            'theta': option.theta,
                            'vega': option.vega,
                            'saved_at': datetime.now()
                        }
                        all_options.append(option_dict)
            
            if all_options:
                df = pd.DataFrame(all_options)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved options chain: {filepath} ({len(all_options)} options)")
                return str(filepath)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error saving options chain for {symbol}: {e}")
            return ""
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """Log trade details to CSV"""
        try:
            filepath = self.base_dir / "trades" / "trade_log.csv"
            
            # Prepare trade record
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': trade_data.get('symbol', ''),
                'option_symbol': trade_data.get('option_symbol', ''),
                'action': trade_data.get('action', ''),
                'quantity': trade_data.get('quantity', 0),
                'price': trade_data.get('price', 0),
                'strike': trade_data.get('strike', 0),
                'expiry': trade_data.get('expiry', ''),
                'option_type': trade_data.get('option_type', ''),
                'strategy': trade_data.get('strategy', ''),
                'confidence': trade_data.get('confidence', 0),
                'reasoning': trade_data.get('reasoning', ''),
                'pnl': trade_data.get('pnl', 0),
                'status': trade_data.get('status', 'open')
            }
            
            # Check if file exists
            file_exists = filepath.exists()
            
            # Write to CSV
            with open(filepath, 'a', newline='') as csvfile:
                fieldnames = trade_record.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(trade_record)
            
            logger.info(f"Logged trade: {trade_data.get('action')} {trade_data.get('option_symbol')}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return ""
    
    def save_analysis_results(self, symbol: str, analysis: Dict[str, Any]) -> str:
        """Save market analysis results to JSON and CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as JSON
            json_filename = f"{symbol}_analysis_{timestamp}.json"
            json_filepath = self.base_dir / "analysis" / json_filename
            
            analysis_copy = analysis.copy()
            analysis_copy['timestamp'] = datetime.now().isoformat()
            analysis_copy['symbol'] = symbol
            
            with open(json_filepath, 'w') as f:
                json.dump(analysis_copy, f, indent=2, default=str)
            
            # Save key metrics to CSV for easy analysis
            csv_filepath = self.base_dir / "analysis" / "analysis_log.csv"
            
            analysis_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'current_price': analysis.get('current_price', 0),
                'daily_change': analysis.get('daily_change', 0),
                'daily_trend': analysis.get('daily_trend', {}).get('trend', ''),
                'daily_strength': analysis.get('daily_trend', {}).get('strength', ''),
                'minute_trend': analysis.get('minute_trend', {}).get('trend', ''),
                'recent_gaps': analysis.get('recent_gaps', 0),
                'latest_gap': analysis.get('latest_gap', 0),
                'volatility': analysis.get('volatility', 0),
                'ml_signal': analysis.get('ml_prediction', {}).get('signal', 0),
                'ml_confidence': analysis.get('ml_prediction', {}).get('confidence', 0),
                'rsi_condition': analysis.get('daily_trend', {}).get('rsi_condition', ''),
                'bullish_engulfing': analysis.get('daily_patterns', {}).get('bullish_engulfing', False),
                'bearish_engulfing': analysis.get('daily_patterns', {}).get('bearish_engulfing', False)
            }
            
            file_exists = csv_filepath.exists()
            
            with open(csv_filepath, 'a', newline='') as csvfile:
                fieldnames = analysis_record.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(analysis_record)
            
            logger.info(f"Saved analysis results: {json_filepath}")
            return str(json_filepath)
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return ""
    
    def create_backtest_log(self, backtest_id: str, config: Dict[str, Any]) -> str:
        """Create a new backtest log file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{backtest_id}_{timestamp}.csv"
            filepath = self.base_dir / "backtest" / filename
            
            # Create backtest metadata file
            metadata_file = self.base_dir / "backtest" / f"backtest_{backtest_id}_{timestamp}_metadata.json"
            metadata = {
                'backtest_id': backtest_id,
                'created_at': datetime.now().isoformat(),
                'config': config,
                'log_file': str(filepath)
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Create CSV with headers
            headers = [
                'timestamp', 'symbol', 'action', 'option_symbol', 'strike', 'expiry',
                'option_type', 'quantity', 'entry_price', 'exit_price', 'pnl',
                'strategy', 'confidence', 'reasoning', 'days_held', 'max_profit',
                'max_loss', 'exit_reason'
            ]
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
            
            logger.info(f"Created backtest log: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating backtest log: {e}")
            return ""
    
    def log_backtest_trade(self, backtest_file: str, trade_data: Dict[str, Any]):
        """Log a backtest trade to the specified file"""
        try:
            with open(backtest_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                row = [
                    trade_data.get('timestamp', ''),
                    trade_data.get('symbol', ''),
                    trade_data.get('action', ''),
                    trade_data.get('option_symbol', ''),
                    trade_data.get('strike', 0),
                    trade_data.get('expiry', ''),
                    trade_data.get('option_type', ''),
                    trade_data.get('quantity', 0),
                    trade_data.get('entry_price', 0),
                    trade_data.get('exit_price', 0),
                    trade_data.get('pnl', 0),
                    trade_data.get('strategy', ''),
                    trade_data.get('confidence', 0),
                    trade_data.get('reasoning', ''),
                    trade_data.get('days_held', 0),
                    trade_data.get('max_profit', 0),
                    trade_data.get('max_loss', 0),
                    trade_data.get('exit_reason', '')
                ]
                
                writer.writerow(row)
            
        except Exception as e:
            logger.error(f"Error logging backtest trade: {e}")
    
    def get_trade_summary(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get trade summary from logs"""
        try:
            filepath = self.base_dir / "trades" / "trade_log.csv"
            
            if not filepath.exists():
                return pd.DataFrame()
            
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting trade summary: {e}")
            return pd.DataFrame()
    
    def get_analysis_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get analysis history for a symbol"""
        try:
            filepath = self.base_dir / "analysis" / "analysis_log.csv"
            
            if not filepath.exists():
                return pd.DataFrame()
            
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by symbol and date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = df[(df['symbol'] == symbol) & 
                   (df['timestamp'] >= start_date) & 
                   (df['timestamp'] <= end_date)]
            
            return df.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return pd.DataFrame()
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """Clean up old data files to save space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            directories_to_clean = [
                self.base_dir / "historical",
                self.base_dir / "analysis",
                self.base_dir / "options_chains"
            ]
            
            for directory in directories_to_clean:
                if not directory.exists():
                    continue
                
                for file_path in directory.glob("*.csv"):
                    if file_path.name.endswith("_latest.csv"):
                        continue  # Keep latest files
                    
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def export_performance_report(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> str:
        """Export comprehensive performance report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.base_dir / "reports" / f"performance_report_{timestamp}.csv"
            
            # Create reports directory if it doesn't exist
            report_file.parent.mkdir(exist_ok=True)
            
            # Get trade data
            trades_df = self.get_trade_summary(start_date, end_date)
            
            if trades_df.empty:
                logger.warning("No trade data available for report")
                return ""
            
            # Calculate performance metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = trades_df['pnl'].sum()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Create summary report
            summary = {
                'report_date': datetime.now(),
                'period_start': start_date or trades_df['timestamp'].min(),
                'period_end': end_date or trades_df['timestamp'].max(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_pct': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'max_win': trades_df['pnl'].max(),
                'max_loss': trades_df['pnl'].min()
            }
            
            # Save detailed trades with summary
            with open(report_file, 'w', newline='') as csvfile:
                # Write summary first
                writer = csv.writer(csvfile)
                writer.writerow(['PERFORMANCE SUMMARY'])
                for key, value in summary.items():
                    writer.writerow([key, value])
                
                writer.writerow([])  # Empty row
                writer.writerow(['DETAILED TRADES'])
                
                # Write detailed trades
                trades_df.to_csv(csvfile, index=False)
            
            logger.info(f"Performance report exported: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return ""
