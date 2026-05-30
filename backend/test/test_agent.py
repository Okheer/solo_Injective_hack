import asyncio
import sys
import os
from dotenv import load_dotenv

# Ensure backend folder is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import TradingAgent
from backend.signals import SignalResult

# ANSI colors for premium terminal UI
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

async def main():
    load_dotenv()
    
    config = {
        "market": "INJ/USDT PERP",
        "market_id": "0x9b9980167ecc3645ff1a5517886652d94a0825e54a77d2057cbbe3ebee015963",
        "strategy": "Funding rate carry + Hurst-corrected RSI",
        "size_inj": 0.5,
        "leverage": 2,
        "stop_loss_pct": 1.5,
        "dry_run": True
    }
    
    print(f"\n{BOLD}{CYAN}============================================================={RESET}")
    print(f"{BOLD}{CYAN}      INJECTIVE AI TRADING JOURNAL — AGENT LOOP & AI TEST     {RESET}")
    print(f"{BOLD}{CYAN}============================================================={RESET}")
    
    # 1. Initialize TradingAgent
    print(f"\n{BOLD}[1/3] Initializing TradingAgent in Dry-Run Mode...{RESET}")
    agent = TradingAgent(config)
    print(f"{GREEN}✓ Agent initialized.{RESET} Dry run: {BOLD}{agent.dry_run}{RESET}")
    
    # 2. Construct a Mock Oversold Long Signal
    print(f"\n{BOLD}[2/3] Constructing a Mock Triggering Signal (Oversold Long)...{RESET}")
    mock_signal = SignalResult(
        action="long",
        funding_rate=-0.00124,  # -0.124% per interval (extremely negative funding = high carry for long)
        carry_signal=1,
        rsi=24.50,             # Oversold (< 35)
        hurst_H=0.214,         # Highly mean-reverting anti-persistent
        corrected_period=5,
        confidence="high",
        reasoning_inputs={
            "funding_rate": -0.00124,
            "funding_threshold": 0.001,
            "rsi": 24.50,
            "rsi_oversold_threshold": 35.0,
            "rsi_overbought_threshold": 65.0,
            "hurst_H": 0.214,
            "standard_rsi_period": 14,
            "corrected_rsi_period": 5,
        }
    )
    print(f"  Mock Signal: Action={BOLD}{mock_signal.action.upper()}{RESET} | Funding={mock_signal.funding_rate} | RSI={mock_signal.rsi} | Hurst H={mock_signal.hurst_H}")
    
    # 3. Fire Simulated Order Execution
    print(f"\n{BOLD}[3/3] Firing Simulated Order Execution & AI Reasoning Generation...{RESET}")
    try:
        await agent.execute_trade(mock_signal)
        print(f"\n{GREEN}✓ Execution completed successfully!{RESET}")
        
        # Verify stored trades
        print(f"\n{BOLD}{MAGENTA}--- VERIFYING LOGGED TRADE STATE ---{RESET}")
        print(f"Total trades recorded: {len(agent.trades)}")
        if agent.trades:
            trade = agent.trades[0]
            print(f"• {BOLD}Logged timestamp:{RESET} {trade['timestamp']}")
            print(f"• {BOLD}Simulated Tx Hash:{RESET} {BLUE}{trade['trade_data']['tx_hash']}{RESET}")
            print(f"• {BOLD}Memo built (length={len(trade['memo'])}):{RESET} {YELLOW}{trade['memo']}{RESET}")
            
            print(f"\n• {BOLD}Full Reasoning Log JSON:{RESET}")
            import json
            parsed_log = json.loads(trade['reasoning_log'])
            print(json.dumps(parsed_log, indent=2))
            
            print(f"\n{GREEN}✓ Milestone 2 completely verified and operational!{RESET}")
            
    except Exception as e:
        print(f"{RED}✗ Verification failed: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
