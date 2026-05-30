import asyncio
import sys
sys.path.append('backend')
from signals import fetch_funding_rate, fetch_recent_prices, generate_signal

async def main():
    market_id = "0x06117805048354477c1d920c7565a91ae13956467332025170d55bc7404e1772"
    print("Fetching funding rate...")
    rate = await fetch_funding_rate(market_id)
    print(f"Funding rate: {rate}")
    
    print("Fetching recent prices...")
    prices = await fetch_recent_prices(market_id, 100)
    print(f"Prices count: {len(prices)}")
    if prices:
        print(f"First 5 prices: {prices[:5]}")
        print(f"Last 5 prices: {prices[-5:]}")
        
    print("Generating signal...")
    sig = generate_signal(rate, prices)
    print(f"Signal generated: {sig}")

if __name__ == '__main__':
    asyncio.run(main())
