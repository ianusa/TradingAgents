#!/usr/bin/env python3
"""
Trading Agent Wrapper Script
Designed to be called from Node.js with command line arguments
"""
import sys
import json
import argparse
import os
import traceback
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

def main():
    parser = argparse.ArgumentParser(description='TradingAgents Decision Wrapper')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., NVDA)')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--deep-think-llm', required=True, help='Deep thinking LLM model')
    parser.add_argument('--quick-think-llm', required=True, help='Quick thinking LLM model')
    parser.add_argument('--backend-url', default='https://api.openai.com/v1', help='Backend URL for LLM API')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--online-tools', action='store_true', default=True, help='Use online tools')
    parser.add_argument('--max-debate-rounds', type=int, default=1, help='Maximum debate rounds')

    args = parser.parse_args()

    try:
        # Create a custom config
        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"] = args.deep_think_llm
        config["quick_think_llm"] = args.quick_think_llm
        config["backend_url"] = args.backend_url
        config["max_debate_rounds"] = args.max_debate_rounds
        config["online_tools"] = args.online_tools

        # Auto-detect LLM provider based on backend URL
        backend_url_lower = args.backend_url.lower()
        if "api.openai.com" in backend_url_lower:
            config["llm_provider"] = "openai"
        elif "anthropic.com" in backend_url_lower:
            config["llm_provider"] = "anthropic"
        elif "generativelanguage.googleapis.com" in backend_url_lower:
            config["llm_provider"] = "google"
        elif "ollama" in backend_url_lower or "localhost:11434" in backend_url_lower:
            config["llm_provider"] = "ollama"
        elif "openrouter.ai" in backend_url_lower:
            config["llm_provider"] = "openrouter"
        else:
            raise ValueError(f"Unable to auto-detect LLM provider from backend URL: {args.backend_url}. Supported URLs must contain one of: api.openai.com, anthropic.com, generativelanguage.googleapis.com, ollama, localhost:11434, openrouter.ai")

        # Initialize with custom config
        ta = TradingAgentsGraph(debug=args.debug, config=config)

        # Forward propagate and get decision
        _, decision = ta.propagate(args.symbol, args.date)

        # Return JSON response
        result = {
            "success": True,
            "decision": str(decision),
            "symbol": args.symbol,
            "date": args.date,
            "deep_think_llm": args.deep_think_llm,
            "quick_think_llm": args.quick_think_llm,
            "backend_url": args.backend_url
        }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "symbol": args.symbol,
            "date": args.date
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()