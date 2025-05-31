"""
data_science_agents/config/settings.py - Simple configuration settings
"""
import os

# Model configuration - using environment variables with sensible defaults
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.3"))

# Analysis limits
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100000"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "50"))

# File handling
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
IMAGES_DIR = "Images"

# Cost estimation (per 1K tokens) - simplified for basic cost tracking
TOKEN_COSTS = {
    "gpt-4o-mini": 0.00015,
    "gpt-4": 0.01,
    "gpt-3.5-turbo": 0.0005
}