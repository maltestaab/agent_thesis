"""
data_science_agents/config/settings.py - Simple configuration settings with optimized limits
"""
import os

# Model configuration - using environment variables with sensible defaults
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.5"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.5"))

# Analysis limits - balanced for comprehensive tasks and comparability
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100000"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))  # Orchestrator turns
MAX_TURNS_SINGLE = int(os.getenv("MAX_TURNS_SINGLE", "500"))  # Single agent total
MAX_TURNS_SPECIALIST = int(os.getenv("MAX_TURNS_SPECIALIST", "50"))  # Specialist agents

# File handling
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
IMAGES_DIR = "Images"

# Current token costs per 1M tokens (updated January 2025)
TOKEN_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 3.00, "output": 10.00}, 
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
}

def get_model_cost(model_name: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Calculate cost for model usage"""
    if model_name not in TOKEN_COSTS:
        model_name = "gpt-4o-mini"  # fallback
    
    costs = TOKEN_COSTS[model_name]
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000