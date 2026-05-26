# On-Chain AI Trading Journal — "Why Did I Trade That?"

An AI agent that trades on Injective's perpetual markets and permanently inscribes a signed, human-readable reasoning log on-chain after every single trade — making it the first fully transparent, self-auditing DeFi trading agent.

## 🚀 The Problem
Every DeFi trading bot today is a **black box**. You see the transaction, but you have no idea *why* it traded. There is no audit trail for losses, and you cannot verify if the agent followed its stated strategy.

## 💡 The Solution: The "Glass Box"
This agent solves the principal-agent problem in DeFi by providing:
- **Autonomous Trading:** Monitors Injective perpetual markets in real-time.
- **On-Chain Reasoning:** Automatically writes a structured reasoning log to the Injective blockchain (via transaction memos) after every trade.
- **Self-Audit:** A chat interface where the agent reads its own on-chain history to explain its performance and mistakes.

## 🧠 Core Strategy
The agent uses two quantitatively-grounded signals:
1.  **Funding Rate Carry:** Exploits Fama’s UIP violation in perpetual markets.
2.  **Hurst-Corrected RSI:** Dynamically adjusts RSI lookback based on the Hurst exponent (H ≈ 0.1 for Injective), capturing fast mean-reversion that standard indicators miss.

## 🛠️ Tech Stack
- **Blockchain:** Injective (Testnet/Mainnet)
- **AI:** Anthropic Claude (Reasoning) & Google Gemini (Analysis)
- **Backend:** Python (FastAPI, Injective-py, NumPy)
- **Frontend:** React (Vite, Tailwind CSS)
- **Framework:** Injective MCP Server

## 🏁 Getting Started
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt` and `npm install` in frontend.
3. Set up `.env` with `INJ_PRIVATE_KEY` and `ANTHROPIC_API_KEY`.
4. Run backend: `python backend/main.py`
5. Run frontend: `npm run dev`
