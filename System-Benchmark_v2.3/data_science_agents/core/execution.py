"""
data_science_agents/core/execution.py - Improved execution engine with cleaner code
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import io
import traceback
from agents import function_tool, RunContextWrapper
from data_science_agents.config.settings import IMAGES_DIR


# Configure plotting settings
import warnings
warnings.filterwarnings('ignore')

# Ensure Images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Simple global state for code execution
execution_namespace = {
    'pd': pd,
    'np': np, 
    'stats': stats,
    'plt': plt,
    'sns': sns
}

# List to track created images
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


@function_tool # Decorator to make the function a tool that can be used by the agents
def execute_code(ctx: RunContextWrapper, code: str) -> str:
    """
    Execute Python code with context awareness.
    
    Args:
        ctx: Analysis context for state management  
        code: Python code to execute
        
    Returns:
        Execution results and output
    """
    try:
        # Build clean context info
        context_info = _build_context_info(ctx)
        
        # Track images before execution
        images_before = _get_current_images()
        
        # Execute code with output capture
        output = _execute_with_capture(code)
        
        # Handle new images
        new_images = _track_new_images(images_before)
        
        # Build and return result
        return _format_result(output, new_images, context_info)
        
    except Exception as e:
        return _format_error(e)


def _build_context_info(ctx: RunContextWrapper) -> str:
    """Build context information string with original prompt."""
    if not hasattr(ctx, 'context') or not ctx.context:
        return ""
    
    try:
        context_lines = []
        
        # Add original prompt if available
        if hasattr(ctx.context, 'original_prompt') and ctx.context.original_prompt:
            context_lines.append(f"# ORIGINAL REQUEST: {ctx.context.original_prompt}")
        
        # Add available variables
        available_vars = ctx.context.get_available_variables()
        if available_vars:
            var_names = list(available_vars.keys())
            if len(var_names) <= 3:
                context_lines.append(f"# Available: {', '.join(var_names)}")
            else:
                context_lines.append(f"# Available: {', '.join(var_names[:3])}, +{len(var_names)-3} more")
        
        return '\n'.join(context_lines) + '\n' if context_lines else ""
    
    except Exception:
        return ""


def _get_current_images() -> set:
    """Get current set of images in the Images directory."""
    if not os.path.exists(IMAGES_DIR):
        return set()
    return set(os.listdir(IMAGES_DIR))


def _execute_with_capture(code: str) -> str:
    """Execute code and capture printed output. (CORE FUNCTION!)"""
    # Capture stdout
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    
    try:
        sys.stdout = captured_output
        exec(code, execution_namespace)
        
        # Check if the last line is a variable reference and show its value
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # If last line is just a variable name, show its value
            if (last_line and 
                last_line.isidentifier() and 
                last_line in execution_namespace and 
                not last_line.startswith('_')):
                
                value = execution_namespace[last_line]
                print(f"{last_line} = {repr(value)}")
        
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def _track_new_images(images_before: set) -> list:
    """Track and register new images created during execution."""
    images_after = _get_current_images()
    new_images = sorted(images_after - images_before)
    
    # Update global tracking
    for img in new_images:
        img_path = os.path.join(IMAGES_DIR, img)
        if img_path not in created_images:
            created_images.append(img_path)
    
    return new_images


def _format_result(output: str, new_images: list, context_info: str) -> str:
    result_parts = []
    
    if output.strip():
        result_parts.append(output.strip())
    
    if new_images:
        if len(new_images) == 1:
            result_parts.append(f"Created: {new_images[0]}")
        else:
            result_parts.append(f"Created {len(new_images)} images: {', '.join(new_images)}")
    
    # Better default message when no output
    if result_parts:
        return '\n'.join(result_parts)
    else:
        return "Code executed successfully (no output to display)" 

def _format_error(error: Exception) -> str:
    """Format error messages for better debugging."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Get clean traceback
    tb_lines = traceback.format_exc().split('\n')
    # Filter out internal execution lines, keep only relevant ones
    relevant_tb = []
    for line in tb_lines:
        if 'exec(code, execution_namespace)' in line:
            continue
        if line.strip():
            relevant_tb.append(line)
    
    # Build clean error message
    result = f"Error ({error_type}): {error_msg}"
    
    if len(relevant_tb) > 3:  # Only add traceback if it's useful
        result += f"\n\nTraceback:\n" + '\n'.join(relevant_tb[-3:])  # Last 3 lines
    
    return result


def get_created_images():
    """Get list of all images created during execution"""
    return created_images.copy()


def get_available_variables():
    """Get dictionary of available variables in the execution namespace"""
    user_vars = {}
    core_imports = {'pd', 'np', 'stats', 'plt', 'sns'}
    
    for name, value in execution_namespace.items():
        if name.startswith('_') or name in core_imports:
            continue
            
        try:
            # Get concise variable description
            if hasattr(value, 'shape'):
                user_vars[name] = f"{type(value).__name__}({value.shape})"
            elif hasattr(value, '__len__') and not isinstance(value, str):
                user_vars[name] = f"{type(value).__name__}[{len(value)}]"
            else:
                user_vars[name] = type(value).__name__
        except Exception:
            user_vars[name] = type(value).__name__
    
    return user_vars