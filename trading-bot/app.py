#!/usr/bin/env python3
"""
Desktop App Launcher for Nifty Options Trading Bot
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
import os
from datetime import datetime
import queue
import logging

from core.trading_engine import TradingEngine
from config.config_manager import ConfigManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TradingBotGUI:
    """Desktop GUI for trading bot"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nifty Options Trading Bot v2.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Bot state
        self.trading_engine = None
        self.is_running = False
        self.config = {}
        
        # Message queue for thread communication
        self.message_queue = queue.Queue()
        
        self.setup_gui()
        self.load_config()
        
        # Start message processing
        self.process_messages()
    
    def setup_gui(self):
        """Setup the GUI components"""
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2b2b2b', foreground='white')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#2b2b2b', foreground='white')
        style.configure('Status.TLabel', font=('Arial', 10), background='#2b2b2b', foreground='#00ff00')
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Advanced Nifty Options Trading Bot", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel Tab
        self.setup_control_panel(notebook)
        
        # Configuration Tab
        self.setup_config_panel(notebook)
        
        # Performance Tab
        self.setup_performance_panel(notebook)
        
        # Logs Tab
        self.setup_logs_panel(notebook)
    
    def setup_control_panel(self, notebook):
        """Setup control panel tab"""
        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="🎮 Control Panel")
        
        # Status section
        status_frame = ttk.LabelFrame(control_frame, text="Bot Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="● Stopped", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT)
        
        # Control buttons
        button_frame = ttk.Frame(status_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.start_button = ttk.Button(button_frame, text="🚀 Start Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.backtest_button = ttk.Button(button_frame, text="📊 Run Backtest", command=self.run_backtest)
        self.backtest_button.pack(side=tk.LEFT)
        
        # Market conditions section
        conditions_frame = ttk.LabelFrame(control_frame, text="Current Market Conditions", padding=10)
        conditions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview for market conditions
        columns = ('Symbol', 'Condition', 'Trend', 'Volatility', 'Signal', 'Confidence')
        self.conditions_tree = ttk.Treeview(conditions_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.conditions_tree.heading(col, text=col)
            self.conditions_tree.column(col, width=120)
        
        scrollbar = ttk.Scrollbar(conditions_frame, orient=tk.VERTICAL, command=self.conditions_tree.yview)
        self.conditions_tree.configure(yscrollcommand=scrollbar.set)
        
        self.conditions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Positions section
        positions_frame = ttk.LabelFrame(control_frame, text="Active Positions", padding=10)
        positions_frame.pack(fill=tk.BOTH, expand=True)
        
        pos_columns = ('Symbol', 'Option', 'Type', 'Qty', 'Entry', 'Current', 'P&L', 'Status')
        self.positions_tree = ttk.Treeview(positions_frame, columns=pos_columns, show='headings', height=6)
        
        for col in pos_columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        pos_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pos_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_config_panel(self, notebook):
        """Setup configuration panel tab"""
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="⚙️ Configuration")
        
        # API Configuration
        api_frame = ttk.LabelFrame(config_frame, text="Fyers API Configuration", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(api_frame, text="Client ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.client_id_entry = ttk.Entry(api_frame, width=40)
        self.client_id_entry.grid(row=0, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(api_frame, text="Secret Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.secret_key_entry = ttk.Entry(api_frame, width=40, show="*")
        self.secret_key_entry.grid(row=1, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(api_frame, text="Access Token:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.access_token_entry = ttk.Entry(api_frame, width=40, show="*")
        self.access_token_entry.grid(row=2, column=1, padx=(10, 0), pady=2)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(config_frame, text="Risk Management", padding=10)
        risk_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(risk_frame, text="Max Daily Loss:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_loss_entry = ttk.Entry(risk_frame, width=20)
        self.max_loss_entry.grid(row=0, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(risk_frame, text="Max Positions:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.max_positions_entry = ttk.Entry(risk_frame, width=20)
        self.max_positions_entry.grid(row=0, column=3, padx=(10, 0), pady=2)
        
        ttk.Label(risk_frame, text="Stop Loss %:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.stop_loss_entry = ttk.Entry(risk_frame, width=20)
        self.stop_loss_entry.grid(row=1, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(risk_frame, text="Profit Target %:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.profit_target_entry = ttk.Entry(risk_frame, width=20)
        self.profit_target_entry.grid(row=1, column=3, padx=(10, 0), pady=2)
        
        # Trading Settings
        trading_frame = ttk.LabelFrame(config_frame, text="Trading Settings", padding=10)
        trading_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(trading_frame, text="Symbols:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.symbols_entry = ttk.Entry(trading_frame, width=40)
        self.symbols_entry.grid(row=0, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(trading_frame, text="Cycle Interval (min):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cycle_interval_entry = ttk.Entry(trading_frame, width=20)
        self.cycle_interval_entry.grid(row=1, column=1, padx=(10, 0), pady=2)
        
        # Buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="💾 Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🔄 Load Config", command=self.load_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🧪 Test API", command=self.test_api).pack(side=tk.LEFT)
    
    def setup_performance_panel(self, notebook):
        """Setup performance panel tab"""
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="📈 Performance")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(perf_frame, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create metrics display
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(metrics_grid, text="Total P&L:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.total_pnl_label = ttk.Label(metrics_grid, text="₹0.00", foreground='green')
        self.total_pnl_label.grid(row=0, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(metrics_grid, text="Today's P&L:", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.daily_pnl_label = ttk.Label(metrics_grid, text="₹0.00", foreground='green')
        self.daily_pnl_label.grid(row=0, column=3, padx=(10, 0), pady=2)
        
        # Row 2
        ttk.Label(metrics_grid, text="Win Rate:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.win_rate_label = ttk.Label(metrics_grid, text="0%")
        self.win_rate_label.grid(row=1, column=1, padx=(10, 0), pady=2)
        
        ttk.Label(metrics_grid, text="Total Trades:", font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.total_trades_label = ttk.Label(metrics_grid, text="0")
        self.total_trades_label.grid(row=1, column=3, padx=(10, 0), pady=2)
        
        # Market condition performance
        condition_perf_frame = ttk.LabelFrame(perf_frame, text="Performance by Market Condition", padding=10)
        condition_perf_frame.pack(fill=tk.BOTH, expand=True)
        
        perf_columns = ('Condition', 'Trades', 'Win Rate', 'Avg P&L', 'Best Strategy')
        self.perf_tree = ttk.Treeview(condition_perf_frame, columns=perf_columns, show='headings', height=12)
        
        for col in perf_columns:
            self.perf_tree.heading(col, text=col)
            self.perf_tree.column(col, width=150)
        
        perf_scrollbar = ttk.Scrollbar(condition_perf_frame, orient=tk.VERTICAL, command=self.perf_tree.yview)
        self.perf_tree.configure(yscrollcommand=perf_scrollbar.set)
        
        self.perf_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_logs_panel(self, notebook):
        """Setup logs panel tab"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="📝 Logs")
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=30, bg='#1e1e1e', fg='#00ff00', font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(log_controls, text="🔄 Refresh", command=self.refresh_logs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_controls, text="🗑️ Clear", command=self.clear_logs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_controls, text="💾 Export", command=self.export_logs).pack(side=tk.LEFT)
    
    def start_bot(self):
        """Start the trading bot"""
        try:
            if not self.validate_config():
                return
            
            self.trading_engine = TradingEngine(self.config)
            
            # Start bot in separate thread
            self.bot_thread = threading.Thread(target=self.run_bot, daemon=True)
            self.bot_thread.start()
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="● Running", foreground='#00ff00')
            
            self.log_message("🚀 Trading bot started successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start bot: {e}")
            logger.error(f"Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the trading bot"""
        try:
            if self.trading_engine:
                self.trading_engine.stop()
            
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="● Stopped", foreground='#ff0000')
            
            self.log_message("⏹️ Trading bot stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop bot: {e}")
            logger.error(f"Failed to stop bot: {e}")
    
    def run_bot(self):
        """Run the trading bot (in separate thread)"""
        try:
            self.trading_engine.start()
        except Exception as e:
            self.message_queue.put(('error', f"Bot error: {e}"))
    
    def run_backtest(self):
        """Run backtesting"""
        try:
            from backtest_engine import BacktestEngine
            
            backtest_engine = BacktestEngine(self.config)
            
            # Run backtest in separate thread
            def run_backtest_thread():
                try:
                    results = backtest_engine.run_backtest()
                    self.message_queue.put(('backtest_complete', results))
                except Exception as e:
                    self.message_queue.put(('error', f"Backtest error: {e}"))
            
            threading.Thread(target=run_backtest_thread, daemon=True).start()
            self.log_message("📊 Starting backtest...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run backtest: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        if not self.client_id_entry.get():
            messagebox.showerror("Error", "Client ID is required")
            return False
        
        if not self.secret_key_entry.get():
            messagebox.showerror("Error", "Secret Key is required")
            return False
        
        if not self.access_token_entry.get():
            messagebox.showerror("Error", "Access Token is required")
            return False
        
        return True
    
    def save_config(self):
        """Save configuration"""
        try:
            config = {
                'fyers': {
                    'client_id': self.client_id_entry.get(),
                    'secret_key': self.secret_key_entry.get(),
                    'redirect_uri': 'https://trade.fyers.in/api-login/redirect-to-app',
                    'access_token': self.access_token_entry.get()
                },
                'risk_management': {
                    'max_daily_loss': float(self.max_loss_entry.get() or 10000),
                    'max_positions': int(self.max_positions_entry.get() or 3),
                    'stop_loss_pct': float(self.stop_loss_entry.get() or 200),
                    'profit_target_pct': float(self.profit_target_entry.get() or 50)
                },
                'trading': {
                    'symbols': self.symbols_entry.get().split(',') if self.symbols_entry.get() else ['NIFTY', 'BANKNIFTY'],
                    'cycle_interval': int(self.cycle_interval_entry.get() or 5)
                }
            }
            
            with open('config/config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            self.config = config
            messagebox.showinfo("Success", "Configuration saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
    
    def load_config(self):
        """Load configuration"""
        try:
            if os.path.exists('config/config.json'):
                with open('config/config.json', 'r') as f:
                    config = json.load(f)
                
                # Populate GUI fields
                fyers_config = config.get('fyers', {})
                self.client_id_entry.delete(0, tk.END)
                self.client_id_entry.insert(0, fyers_config.get('client_id', ''))
                
                self.secret_key_entry.delete(0, tk.END)
                self.secret_key_entry.insert(0, fyers_config.get('secret_key', ''))
                
                self.access_token_entry.delete(0, tk.END)
                self.access_token_entry.insert(0, fyers_config.get('access_token', ''))
                
                risk_config = config.get('risk_management', {})
                self.max_loss_entry.delete(0, tk.END)
                self.max_loss_entry.insert(0, str(risk_config.get('max_daily_loss', 10000)))
                
                self.max_positions_entry.delete(0, tk.END)
                self.max_positions_entry.insert(0, str(risk_config.get('max_positions', 3)))
                
                self.stop_loss_entry.delete(0, tk.END)
                self.stop_loss_entry.insert(0, str(risk_config.get('stop_loss_pct', 200)))
                
                self.profit_target_entry.delete(0, tk.END)
                self.profit_target_entry.insert(0, str(risk_config.get('profit_target_pct', 50)))
                
                trading_config = config.get('trading', {})
                self.symbols_entry.delete(0, tk.END)
                self.symbols_entry.insert(0, ','.join(trading_config.get('symbols', ['NIFTY', 'BANKNIFTY'])))
                
                self.cycle_interval_entry.delete(0, tk.END)
                self.cycle_interval_entry.insert(0, str(trading_config.get('cycle_interval', 5)))
                
                self.config = config
                
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def test_api(self):
        """Test API connection"""
        try:
            from api.fyers_client import FyersClient
            
            client = FyersClient(
                client_id=self.client_id_entry.get(),
                secret_key=self.secret_key_entry.get(),
                redirect_uri='https://trade.fyers.in/api-login/redirect-to-app',
                access_token=self.access_token_entry.get()
            )
            
            if client.authenticate():
                messagebox.showinfo("Success", "✅ API connection successful!")
            else:
                messagebox.showerror("Error", "❌ API connection failed!")
                
        except Exception as e:
            messagebox.showerror("Error", f"API test failed: {e}")
    
    def log_message(self, message: str):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    def refresh_logs(self):
        """Refresh log display"""
        # Implementation to read from log files
        pass
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def export_logs(self):
        """Export logs to file"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Logs exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")
    
    def process_messages(self):
        """Process messages from background threads"""
        try:
            while True:
                message_type, data = self.message_queue.get_nowait()
                
                if message_type == 'error':
                    self.log_message(f"❌ {data}")
                elif message_type == 'backtest_complete':
                    self.log_message("✅ Backtest completed successfully!")
                    # Update performance display with results
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(1000, self.process_messages)
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """Main function"""
    # Create necessary directories
    os.makedirs('config', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Start GUI
    app = TradingBotGUI()
    app.run()

if __name__ == "__main__":
    main()
