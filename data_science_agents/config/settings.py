"""
Configuration settings for the data science agents
"""

# Model settings
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 1.0
MAX_TOKENS = 2000
MAX_TURNS = 5

# File settings
IMAGES_DIR = "Images" # Image Folder
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"] # Supported file types for data loading