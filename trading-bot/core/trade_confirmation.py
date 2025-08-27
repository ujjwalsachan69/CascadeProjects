{{ ... }}

@dataclass
class ExitProposal:
    """Exit proposal data structure"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    quantity: int
    unrealized_pnl: float
    realized_pnl: float
    exit_reason: str
    exit_type: str  # 'PROFIT_TARGET', 'STOP_LOSS', 'MANUAL', 'TIME_BASED'
    points_moved: float
    percentage_change: float
    hold_duration: str
    exit_reasoning: Dict[str, Any]
    timestamp: datetime

class TradeConfirmationSystem:
    """Manual trade confirmation system with detailed analysis"""
    
    def __init__(self):
        self.daily_trades_file = 'data/daily_trades.json'
        self.confirmation_history_file = 'data/confirmation_history.json'
        self.exit_history_file = 'data/exit_history.json'
        self.ensure_data_files()
    
    def ensure_data_files(self):
        """Ensure data files exist"""
        os.makedirs('data', exist_ok=True)
        
        for file_path in [self.daily_trades_file, self.confirmation_history_file, self.exit_history_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump({} if 'daily_trades' in file_path else [], f)

{{ ... }}

    def create_exit_proposal(self, position: Dict[str, Any], current_price: float, 
                           exit_reason: str, exit_type: str) -> ExitProposal:
        """Create an exit proposal with detailed P&L analysis"""
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        direction = position['direction']
        entry_time = datetime.fromisoformat(position['timestamp'])
        
        # Calculate P&L
        if direction.upper() == 'BUY':
            unrealized_pnl = (current_price - entry_price) * quantity
            points_moved = current_price - entry_price
        else:  # SELL
            unrealized_pnl = (entry_price - current_price) * quantity
            points_moved = entry_price - current_price
        
        percentage_change = (points_moved / entry_price) * 100
        hold_duration = str(datetime.now() - entry_time).split('.')[0]  # Remove microseconds
        
        # Create detailed exit reasoning
        exit_reasoning = {
            'trigger_reason': exit_reason,
            'exit_type': exit_type,
            'current_market_time': datetime.now().strftime('%H:%M:%S'),
            'position_age': hold_duration,
            'price_movement': f"{points_moved:+.2f} points ({percentage_change:+.2f}%)",
            'target_hit': current_price >= position.get('profit_target', 0) if direction == 'BUY' else current_price <= position.get('profit_target', 0),
            'stop_hit': current_price <= position.get('stop_loss', 0) if direction == 'BUY' else current_price >= position.get('stop_loss', 0),
            'risk_reward_achieved': abs(unrealized_pnl) / abs(position.get('risk_amount', 1000)) if position.get('risk_amount') else 'N/A'
        }
        
        return ExitProposal(
            trade_id=position['trade_id'],
            symbol=position['symbol'],
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            quantity=quantity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=unrealized_pnl,  # Will be realized upon exit
            exit_reason=exit_reason,
            exit_type=exit_type,
            points_moved=points_moved,
            percentage_change=percentage_change,
            hold_duration=hold_duration,
            exit_reasoning=exit_reasoning,
            timestamp=datetime.now()
        )
    
    def display_exit_analysis(self, proposal: ExitProposal) -> None:
        """Display detailed exit analysis for user review"""
        
        profit_color = "🟢" if proposal.unrealized_pnl >= 0 else "🔴"
        pnl_status = "PROFIT" if proposal.unrealized_pnl >= 0 else "LOSS"
        
        print("\n" + "="*80)
        print(f"🚨 EXIT CONFIRMATION REQUIRED - {pnl_status} {profit_color}")
        print("="*80)
        
        print(f"\n📊 POSITION DETAILS:")
        print(f"   Trade ID: {proposal.trade_id}")
        print(f"   Symbol: {proposal.symbol}")
        print(f"   Direction: {proposal.direction}")
        print(f"   Quantity: {proposal.quantity}")
        print(f"   Hold Duration: {proposal.hold_duration}")
        
        print(f"\n💰 PRICE & P&L ANALYSIS:")
        print(f"   Entry Price: ₹{proposal.entry_price:,.2f}")
        print(f"   Current Price: ₹{proposal.current_price:,.2f}")
        print(f"   Points Moved: {proposal.points_moved:+.2f}")
        print(f"   Percentage Change: {proposal.percentage_change:+.2f}%")
        print(f"   Unrealized P&L: ₹{proposal.unrealized_pnl:+,.2f} {profit_color}")
        
        print(f"\n🎯 EXIT TRIGGER ANALYSIS:")
        print(f"   Exit Reason: {proposal.exit_reason}")
        print(f"   Exit Type: {proposal.exit_type}")
        
        print(f"\n🔍 DETAILED EXIT REASONING:")
        for key, value in proposal.exit_reasoning.items():
            display_key = key.replace('_', ' ').title()
            print(f"   {display_key}: {value}")
        
        # Show exit recommendation
        if proposal.exit_type == 'PROFIT_TARGET':
            print(f"\n✅ RECOMMENDATION: Take profit - Target achieved!")
        elif proposal.exit_type == 'STOP_LOSS':
            print(f"\n⚠️  RECOMMENDATION: Cut losses - Stop loss triggered!")
        elif proposal.unrealized_pnl > 0:
            print(f"\n💡 RECOMMENDATION: Consider taking profit")
        else:
            print(f"\n⚠️  RECOMMENDATION: Consider cutting losses")
        
        print("\n" + "="*80)
    
    def get_exit_confirmation(self, proposal: ExitProposal) -> bool:
        """Get user confirmation for position exit"""
        
        # Display exit analysis
        self.display_exit_analysis(proposal)
        
        # Get user input
        pnl_status = "PROFIT" if proposal.unrealized_pnl >= 0 else "LOSS"
        print(f"\n⚡ Type 'y' to CLOSE position (realize {pnl_status}) or 'n' to KEEP holding:")
        print(f"   Current P&L: ₹{proposal.unrealized_pnl:+,.2f}")
        
        try:
            user_input = input("Your decision (y/n): ").strip().lower()
            
            if user_input == 'y':
                self.record_exit_decision(proposal, True, "User confirmed exit")
                print(f"✅ EXIT CONFIRMED - Closing position...")
                print(f"💰 Realizing P&L: ₹{proposal.unrealized_pnl:+,.2f}")
                return True
            else:
                self.record_exit_decision(proposal, False, f"User rejected exit: {user_input}")
                print("❌ EXIT REJECTED - Keeping position open")
                return False
                
        except Exception as e:
            logger.error(f"Error getting exit confirmation: {e}")
            self.record_exit_decision(proposal, False, f"Error: {e}")
            print("❌ EXIT REJECTED due to error")
            return False
    
    def record_exit_decision(self, proposal: ExitProposal, approved: bool, reason: str):
        """Record the exit decision for history"""
        
        try:
            with open(self.exit_history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
        
        history.append({
            'trade_id': proposal.trade_id,
            'exit_timestamp': proposal.timestamp.isoformat(),
            'symbol': proposal.symbol,
            'direction': proposal.direction,
            'entry_price': proposal.entry_price,
            'exit_price': proposal.current_price,
            'quantity': proposal.quantity,
            'unrealized_pnl': proposal.unrealized_pnl,
            'points_moved': proposal.points_moved,
            'percentage_change': proposal.percentage_change,
            'hold_duration': proposal.hold_duration,
            'exit_reason': proposal.exit_reason,
            'exit_type': proposal.exit_type,
            'approved': approved,
            'decision_reason': reason
        })
        
        with open(self.exit_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def check_exit_conditions(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str, str]:
        """Check if position meets exit conditions"""
        
        entry_price = position['entry_price']
        direction = position['direction']
        stop_loss = position.get('stop_loss', 0)
        profit_target = position.get('profit_target', 0)
        
        # Check stop loss
        if direction.upper() == 'BUY' and current_price <= stop_loss:
            return True, "Stop loss triggered", "STOP_LOSS"
        elif direction.upper() == 'SELL' and current_price >= stop_loss:
            return True, "Stop loss triggered", "STOP_LOSS"
        
        # Check profit target
        if direction.upper() == 'BUY' and current_price >= profit_target:
            return True, "Profit target achieved", "PROFIT_TARGET"
        elif direction.upper() == 'SELL' and current_price <= profit_target:
            return True, "Profit target achieved", "PROFIT_TARGET"
        
        # Check time-based exit (e.g., end of day)
        current_time = datetime.now().time()
        market_close = datetime.strptime('15:20', '%H:%M').time()  # 10 minutes before close
        
        if current_time >= market_close:
            return True, "Market closing - time-based exit", "TIME_BASED"
        
        return False, "", ""
    
    def should_suggest_manual_exit(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """Check if we should suggest manual exit based on various factors"""
        
        entry_price = position['entry_price']
        direction = position['direction']
        entry_time = datetime.fromisoformat(position['timestamp'])
        hold_duration = datetime.now() - entry_time
        
        # Calculate current P&L
        if direction.upper() == 'BUY':
            pnl = (current_price - entry_price) * position['quantity']
            points_moved = current_price - entry_price
        else:
            pnl = (entry_price - current_price) * position['quantity']
            points_moved = entry_price - current_price
        
        # Suggest exit if significant profit (75% of target)
        target_points = abs(position.get('profit_target', entry_price) - entry_price)
        if abs(points_moved) >= target_points * 0.75 and pnl > 0:
            return True, f"Significant profit achieved ({points_moved:+.0f} points)"
        
        # Suggest exit if significant loss (50% of stop loss)
        stop_points = abs(position.get('stop_loss', entry_price) - entry_price)
        if abs(points_moved) >= stop_points * 0.5 and pnl < 0:
            return True, f"Approaching stop loss ({points_moved:+.0f} points)"
        
        # Suggest exit if held for too long without movement
        if hold_duration.total_seconds() > 14400 and abs(pnl) < 500:  # 4 hours, minimal P&L
            return True, "Position stagnant for extended period"
        
        return False, ""

{{ ... }}
