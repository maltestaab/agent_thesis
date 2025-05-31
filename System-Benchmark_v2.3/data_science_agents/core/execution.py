"""
data_science_agents/core/execution.py - Execution engine with context support
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import io
import re
import traceback
from typing import Any, Dict
from agents import function_tool, RunContextWrapper

# Configure plotting settings
import warnings
warnings.filterwarnings('ignore')

# Ensure Images directory exists
IMAGES_DIR = "Images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Simple global state for code execution - much simpler than complex classes
execution_namespace = {
    'pd': pd,
    'np': np, 
    'stats': stats,
    'plt': plt,
    'sns': sns
}

# List to track created images - simple list instead of complex tracking
created_images = []


def reset_execution_state():
    """Reset the execution environment to a clean state"""
    global execution_namespace, created_images
    
    # Clear everything except core imports
    execution_namespace.clear()
    execution_namespace.update({
        'pd': pd,
        'np': np,
        'stats': stats, 
        'plt': plt,
        'sns': sns
    })
    
    # Clear image tracking
    created_images.clear()


def detect_saved_csv_filename(code: str) -> str | None:
    """Simple CSV file detection for df.to_csv(...) calls."""
    match = re.search(r'\.to_csv\(\s*["\'](.+?\.csv)["\']', code)
    return match.group(1) if match else None


@function_tool
def execute_code(ctx: RunContextWrapper, code: str) -> str:
    """
    Execute Python code with context awareness to avoid redundant work.
    
    This tool:
    1. Can access previous work via the context to avoid re-execution
    2. Executes provided code in a shared namespace
    3. Captures any printed output
    4. Tracks any images saved to the Images/ folder
    5. Returns the output and image information
    
    Args:
        ctx: Context wrapper containing analysis state and previous results
        code: Python code to execute
        
    Returns:
        String containing execution results and any output
    """
    try:
        # Access context if available (some agents might not use context)
        context_info = ""
        if hasattr(ctx, 'context') and ctx.context:
            # Provide helpful context information in code comments
            available_vars = ctx.context.get_available_variables()
            if available_vars:
                var_list = ", ".join(available_vars.keys())
                context_info = f"# Available variables: {var_list}\n"
            
            # Add previous results info if this is multi-agent
            if ctx.context.analysis_type == "multi_agent" and ctx.context.agent_results:
                context_info += f"# Previous phases completed: {', '.join(ctx.context.completed_phases)}\n"
        
        # Track images before execution to see what's new
        images_before = set()
        if os.path.exists(IMAGES_DIR):
            images_before = set(os.listdir(IMAGES_DIR))
        
        # Capture printed output using string buffer
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Execute the code in our shared namespace
            # Add context info as comments so model can see what's available
            full_code = context_info + code if context_info else code
            exec(full_code, execution_namespace)  # Execute original code, not with comments
            
            # Get any printed output
            output = captured_output.getvalue()

        finally:
            # Always restore stdout
            sys.stdout = old_stdout
        
        # Check for new images created
        images_after = set()
        if os.path.exists(IMAGES_DIR):
            images_after = set(os.listdir(IMAGES_DIR))
        
        new_images = images_after - images_before
        
        # Update our global image tracking
        for img in sorted(new_images):
            img_path = os.path.join(IMAGES_DIR, img)
            if img_path not in created_images:
                created_images.append(img_path)
        
        # Build result message
        result_parts = []
        
        # Add context info if helpful
        if context_info:
            result_parts.append("Context information was available to guide execution.")
        
        if output.strip():
            result_parts.append(output.strip())
        
        if new_images:
            result_parts.append(f"Created {len(new_images)} new image(s): {', '.join(sorted(new_images))}")
        
        return '\n'.join(result_parts) if result_parts else "Code executed successfully (no output)"
        
    except Exception as e:
        # Simple error handling - just return the error message and traceback
        error_msg = f"Error: {str(e)}"
        tb = traceback.format_exc()
        return f"{error_msg}\n\nTraceback:\n{tb}"


def get_created_images():
    """Get list of all images created during execution"""
    return created_images.copy()


def get_available_variables():
    """Get dictionary of available variables in the execution namespace"""
    # Return only user-created variables (not the core imports)
    user_vars = {}
    for name, value in execution_namespace.items():
        if not name.startswith('_') and name not in ['pd', 'np', 'stats', 'plt', 'sns']:
            try:
                # Get a simple description of the variable
                if hasattr(value, 'shape'):
                    user_vars[name] = f"{type(value).__name__} with shape {value.shape}"
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    user_vars[name] = f"{type(value).__name__} with {len(value)} items"
                else:
                    user_vars[name] = type(value).__name__
            except:
                user_vars[name] = type(value).__name__
    
    return user_vars