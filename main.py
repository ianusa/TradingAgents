from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import time

load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"  # Use a different model
config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # Use a different backend
config["deep_think_llm"] = "gemini-2.5-flash"  # Use a different model
config["quick_think_llm"] = "gemini-2.5-flash"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Increase debate rounds
config["quick_think_llm_thinking_budget"] = 1024
config["quick_think_llm_max_tokens"] = 5120

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
start_time = time.time()
_, decision = ta.propagate("SPY", "2025-07-30")
end_time = time.time()
processing_time = end_time - start_time

minutes = int(processing_time // 60)
seconds = processing_time % 60

print(f"Processing time: {minutes} minutes {seconds:.2f} seconds")
print(f"Decision: {decision}")

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
