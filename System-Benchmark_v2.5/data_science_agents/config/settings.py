"""
data_science_agents/config/settings.py - System Configuration and Pricing

This module centralizes all configuration settings for the data science agent system.
It provides a single place to manage model settings, analysis limits, file handling,
and cost calculations. Think of it as the "control panel" for the entire system.

Key Configuration Areas:
- AI Model Settings: Default models, temperature, and other parameters
- Analysis Limits: Turn budgets and execution constraints  
- File Handling: Supported formats and directory settings
- Cost Tracking: Current pricing for different AI models
- Environment Integration: Reading settings from environment variables

Purpose: This centralized configuration approach ensures:
- Consistent settings across the entire system
- Easy adjustment of system parameters
- Transparent cost calculation based on real pricing
- Environment-specific customization via environment variables
- Clear separation of configuration from business logic

The settings support both development and production environments, with
sensible defaults that can be overridden via environment variables.
"""
import os

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================
# These settings control which AI models are used and how they behave during analysis

# Default AI model used when no specific model is requested
# gpt-4o-mini provides the best balance of capability and cost for most data analysis tasks
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# Model creativity and consistency settings
# Temperature controls randomness: 0.0 = very consistent, 1.0 = very creative
# 0.5 provides a good balance for data analysis (consistent but not rigid)
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.5"))

# Top-p controls diversity of token selection during generation
# 0.5 focuses on the most likely tokens while allowing some variation
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.5"))


# =============================================================================
# ANALYSIS EXECUTION LIMITS
# =============================================================================
# These settings control how much work agents can do, preventing runaway costs and timeouts

# Maximum conversation turns for the orchestrator agent in multi-agent mode
# Each turn includes reasoning + tool calls, so 10 turns allows substantial coordination
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

# Maximum conversation turns for single agent analysis
# Single agents need more turns since they handle all phases themselves
# 500 turns allows comprehensive analysis while preventing infinite loops
MAX_TURNS_SINGLE = int(os.getenv("MAX_TURNS_SINGLE", "500"))

# Maximum conversation turns for specialist agents in multi-agent mode
# Specialists focus on specific phases, so 50 turns is sufficient for most tasks
MAX_TURNS_SPECIALIST = int(os.getenv("MAX_TURNS_SPECIALIST", "50"))


# =============================================================================
# FILE HANDLING CONFIGURATION
# =============================================================================
# Settings for data file processing and output management

# File types that the system can process for analysis
# Currently supports the most common data formats in business and research
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]

# Directory where agent-created images (plots, charts) are saved
# This provides a consistent location for all visualizations
IMAGES_DIR = "Images"


# =============================================================================
# AI MODEL PRICING (CURRENT AS OF JUNE 2025)
# =============================================================================
# Cost per 1 million tokens for different OpenAI models
# These prices are used for real-time cost estimation during analysis

TOKEN_COSTS = {
    # GPT-4o Mini: Reliable and cost-effective for testing and general use
    "gpt-4o-mini": {
        "input": 0.15,   # Cost per 1M input tokens - reliable model good for testing
        "output": 0.60   # Well-established performance with 128K context window
    },
    
    # GPT-4.1 Series: General-purpose models with 1M token context
    "gpt-4.1-mini": {
        "input": 0.40,   # Cost per 1M input tokens - balanced performance and cost
        "output": 1.60   # Nearly matches full GPT-4.1 performance at lower cost
    },
    
    "gpt-4.1-nano": {
        "input": 0.10,   # Cost per 1M input tokens - fastest and most economical
        "output": 0.40   # Ideal for classification, autocompletion, and high-volume tasks
    },
    
    # o-Series: Advanced reasoning models with tool use capabilities
    "o3": {
        "input": 10.00,  # Cost per 1M input tokens - most advanced reasoning model
        "output": 40.00  # Excels at complex reasoning, math, coding, and science
    },
    
    "o3-mini": {
        "input": 1.10,   # Cost per 1M input tokens - advanced model for STEM and reasoning tasks
        "output": 4.40   # Flexible reasoning levels (low/medium/high), great coding performance
    },
    
    "o4-mini": {
        "input": 1.10,   # Cost per 1M input tokens - cost-effective reasoning
        "output": 4.40   # Strong reasoning performance at 10x lower cost than o3
    }
}



def get_model_cost(model_name: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """
    Calculate the total cost for using a specific AI model based on token usage.
    
    This function provides accurate, real-time cost estimation by applying the
    current pricing structure to actual token consumption. Different models have
    very different costs, and input/output tokens are priced differently.
    
    Args:
        model_name (str): Name of the AI model being used (e.g., "gpt-4o-mini")
        input_tokens (int): Number of input tokens consumed (prompts, context, data)
        output_tokens (int): Number of output tokens generated (agent responses, analysis)
        
    Returns:
        float: Total cost in USD for the specified token usage
        
    Cost Calculation:
        - Input and output tokens are priced separately (output typically costs more)
        - Costs are calculated per million tokens, then scaled to actual usage
        - Unknown models default to gpt-4o-mini pricing (most economical fallback)
        
    Example Usage:
        # Calculate cost for a typical analysis session
        cost = get_model_cost("gpt-4o-mini", input_tokens=50000, output_tokens=25000)
        # Returns: (50000 * 0.15 + 25000 * 0.60) / 1,000,000 = $0.0225
        
    Purpose:
        - Provide transparent cost tracking during analysis
        - Enable users to make informed model choices
        - Support budget monitoring and cost optimization
        - Help users understand the value of different models
    """
    # Use fallback pricing if model is not recognized
    if model_name not in TOKEN_COSTS:
        model_name = "gpt-4o-mini"  # Default to most economical option
    
    # Get pricing structure for the specified model
    costs = TOKEN_COSTS[model_name]
    
    # Calculate total cost: (input_tokens * input_rate + output_tokens * output_rate) / 1M
    total_cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
    
    return total_cost



import os
import base64

# Replace with your actual Langfuse keys
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-bc93054c-8067-4c30-9923-ada269c6ec43"  
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-eed57bac-ee32-49cd-9d96-488408c3602c"  
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" 

# Build Basic Auth header
LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()

# Configure OpenTelemetry endpoint & headers
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST") + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
