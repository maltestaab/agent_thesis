"""
data_science_agents/core/execution.py - Enhanced Code Execution Engine

====== WHAT THIS FILE DOES ======
This is the core engine that allows AI agents to run Python code as a tool.

1. **Execute Python Code**: Run data analysis, create visualizations, build models
2. **Maintain State**: Remember variables and data between code executions
3. **Track Results**: Capture outputs, error messages, and created files
4. **Provide Context**: Tell agents what data/variables are already available

====== WHY THIS IS IMPORTANT ======
Without this system, AI agents would be just chatbots that can only talk about data.
With this system, they become actual data scientists that can:
- Load and analyze real datasets
- Create professional visualizations
- Build and evaluate machine learning models
- Generate actionable insights with real numbers

====== KEY CONCEPTS ======
- **Namespace**: A controlled environment where code runs
- **State Persistence**: Variables survive between code executions
- **Context Awareness**: Agents know what data is already loaded
- **Function Tools**: Python functions that AI agents can call
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


# Ignore plotting warnings that sometimes appear when agents create images using specific functions
import warnings
warnings.filterwarnings('ignore')

# Ensure Images directory exists where images created by agents are saved
os.makedirs(IMAGES_DIR, exist_ok=True)

# Simple global state for code execution (namespace). Some preinstalled libraries to avoid errors in the first place.
execution_namespace = {
    'pd': pd,
    'np': np, 
    'stats': stats,
    'plt': plt,
    'sns': sns
}

# List to track created images during an analysis to differentiate between newly created images and images that were already there before the analysis started
created_images = []


def reset_execution_state():
    """Reset the execution environment to a clean state"""
    global execution_namespace, created_images
    
    # Clear everything from namespace except core imports to have independent runs
    execution_namespace.clear()
    execution_namespace.update({
        'pd': pd,
        'np': np,
        'stats': stats, 
        'plt': plt,
        'sns': sns
    })
    
    # Reset image tracking
    created_images.clear()


@function_tool # Decorator to make the function a tool that can be used by the agents (part of Agent SDK)
def execute_code(ctx: RunContextWrapper, code: str) -> str:
    """
    Execute Python code with enhanced context awareness and dataframe continuity.
    This is the main tool that AI agents use to perform data analysis.
    
    Args:
        ctx: Analysis context containing information about current analysis state. 
        code: Python code string to execute (written by the AI agent)
        
    Returns:
        str: Execution results, error messages, or success confirmation
        
    What this function does:
    1. **Context Building**: Tells the agent what data/variables are already available. Avoids the agent reloading the same data multiple times.
    2. **Code Execution**: Running the AI-generated Python code
    3. **Output Capture**: Captures printed output and expression results
    4. **Image Tracking**: Detects and tracks any new visualizations created
    5. **Error Handling**: Provides helpful error messages if something goes wrong
    6. **Result Formatting**: Returns clean, informative results to the agent
    """

    try:
        # Build enhanced context info
        context_info = _build_enhanced_context_info(ctx)
        
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


def _build_enhanced_context_info(ctx: RunContextWrapper) -> str:
    """
    Build enhanced context information with dataframe awareness.
    
    This function creates a "briefing" for the AI agent about what's already
    available in the execution environment. It's like telling a new team member
    "Here's what we've already done and what data is available."
    
    Args:
        ctx: Analysis context wrapper containing current state
        
    Returns:
        str: Context information as comments that get added to code execution
        
    What this provides to agents:
    - Original user request (so they remember the goal)
    - Available dataframes (so they don't reload data unnecessarily)
    - Other variables in memory (models, results, etc.)
    """
    if not hasattr(ctx, 'context') or not ctx.context:
        return ""
    
    try:
        context_lines = []
        
        # Add original prompt if available to remind the agent of the inital user request
        if hasattr(ctx.context, 'original_prompt') and ctx.context.original_prompt:
            context_lines.append(f"# ORIGINAL REQUEST: {ctx.context.original_prompt}")
        
        # Tell the agent what dataframes and variables are already available
        available_vars = ctx.context.get_available_variables()
        if available_vars:
            # Separate dataframes and other variables to make it easier for the agent to understand (since this was an issue during development)
            dataframe_vars = [name for name, desc in available_vars.items() 
                            if 'DataFrame' in desc]
            other_vars = [name for name in available_vars.keys() if name not in dataframe_vars]
            
            if dataframe_vars:
                context_lines.append(f"# AVAILABLE DATAFRAMES: {', '.join(dataframe_vars)} (use these instead of reloading)") # Remind the agent to use the dataframes that are already available instead of reloading them
            
            if other_vars:
                if len(other_vars) <= 3:
                    context_lines.append(f"# OTHER VARIABLES: {', '.join(other_vars)}")
                else:
                    context_lines.append(f"# OTHER VARIABLES: {', '.join(other_vars[:3])}, +{len(other_vars)-3} more")
        
        return '\n'.join(context_lines) + '\n' if context_lines else ""
    
    except Exception:
        return ""


def _get_current_images() -> set:
    """Get filenames of current set of images in the Images directory."""
    if not os.path.exists(IMAGES_DIR):
        return set()
    return set(os.listdir(IMAGES_DIR))


def _execute_with_capture(code: str) -> str:
    """
    Execute code and capture printed output.
    
    This is the core function that actually runs the Python code written by AI agents.
    It's designed to capture both printed output and expression results.
    
    Args:
        code: Python code string to execute
        
    Returns:
        str: Captured output from the code execution
        
    What this function does:
    1. **Output Redirection**: Captures anything printed by the code
    2. **Code Execution**: Runs the code in the global namespace
    3. **Expression Evaluation**: Handles cases where the last line is an expression
    4. **Clean Output**: Returns formatted results
    """
    # Captures outputs
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    
    try:
        sys.stdout = captured_output
        
        # Actually executes the code (!) in the global namespace
        exec(code, execution_namespace)
        
        # Check if the last line is an expression that should show output
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            
            # Checls if the last line of code is an expression rather than a statement
            if last_line and not last_line.startswith('#') and (',' in last_line or last_line.isidentifier()):
                try:
                    # Tries to evaluate the expression and print it. This is similar to how Jupyter notebooks work and make it easier for the agent to see the results.
                    result = eval(last_line, execution_namespace)
                    
                    # Formats the result nicely instead of just showing a tuple
                    if isinstance(result, tuple):
                        print("Results:")
                        for i, item in enumerate(result):
                            print(f"Item {i+1}:")
                            print(item)
                            print("-" * 30)
                    else:
                        print(f"Result: {result}")
                        
                except Exception:
                    # If evaluation fails, just continue
                    pass
        
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def _track_new_images(images_before: set) -> list:
    """Track and register new images created during execution. Necessary for the streamlit app to display the images."""
    images_after = _get_current_images()
    new_images = sorted(images_after - images_before)
    
    # Update global tracking
    for img in new_images:
        img_path = os.path.join(IMAGES_DIR, img)
        if img_path not in created_images:
            created_images.append(img_path)
    
    return new_images


def _format_result(output: str, new_images: list, context_info: str) -> str:
    """
    Format execution results in a clean, informative way.
    
    Args:
        output: Captured output from code execution
        new_images: List of newly created image files
        context_info: Context information (not currently used in output)
        
    Returns:
        str: Formatted result string for the AI agent
        
    This function creates clear, informative feedback for AI agents about
    what happened during code execution. The format helps agents understand:
    - What output was generated
    - What files were created
    - Whether the execution was successful

    --> Improves interaction between agents
    """
    result_parts = []
    
    # Add output if there is any
    if output.strip():
        result_parts.append(output.strip())
    
    # Add images if there are any
    if new_images:
        if len(new_images) == 1:
            result_parts.append(f"Created: {new_images[0]}")
        else:
            result_parts.append(f"Created {len(new_images)} images: {', '.join(new_images)}")
    
    # Better default message when code has no output. Helps the agent understand that the code ran successfully instead of retrying it thinking it failed. 
    if result_parts:
        return '\n'.join(result_parts)
    else:
        return "Code executed successfully (no output to display)" 

def _format_error(error: Exception) -> str:
    """
    Format error messages for better debugging and agent understanding.
    
    Args:
        error: Exception that occurred during code execution
        
    Returns:
        str: Formatted error message with helpful debugging information
        
    This function takes Python errors and makes them more understandable:
    1. **Error Type**: Shows what kind of error occurred
    2. **Error Message**: The specific error description
    3. **Relevant Traceback**: Only the useful parts of the error trace
    
    --> This helps AI agents understand what went wrong and to quickly debug
    """
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
    """
    Get list of all images created during execution
    
    Returns:
        list: Copy of the global created_images list
    """
    return created_images.copy()


def get_available_variables():
    """
    Get dictionary of available variables in the execution namespace.
    
    Returns:
        dict: Variable names mapped to their descriptions
        
    This function provides AI agents with information about what data
    and variables are currently available in memory. This prevents:
    - Unnecessary reloading of data
    - Accidental variable overwrites
    - Confusion about what data is available
    
    Example return:
    {
        'df': 'DataFrame(1000x15)',
        'df_clean': 'DataFrame(985x15)', 
        'model_rf': 'RandomForestRegressor',
        'accuracy_score': 'float'
    }
    """
    user_vars = {}
    # Core imports that should be excluded from the variable list
    core_imports = {'pd', 'np', 'stats', 'plt', 'sns'}
    
    for name, value in execution_namespace.items():
        # Skip private variables and core imports
        if name.startswith('_') or name in core_imports:
            continue
            
        try:
            # Enhanced descriptions for dataframes so the agents can more easily work with them
            if isinstance(value, pd.DataFrame):
                user_vars[name] = f"DataFrame({value.shape[0]}x{value.shape[1]})"
            elif hasattr(value, 'shape'):
                user_vars[name] = f"{type(value).__name__}({value.shape})"
            elif hasattr(value, '__len__') and not isinstance(value, str):
                user_vars[name] = f"{type(value).__name__}[{len(value)}]"
            else:
                user_vars[name] = type(value).__name__
        except Exception:
            user_vars[name] = type(value).__name__
    
    return user_vars


def get_dataframe_variables():
    """
    Get dictionary of only dataframe variables for easy identification.
    
    Returns:
        dict: Detailed information about available DataFrames  
        
    This function provides comprehensive information about DataFrames specifically,
    since they're the most important variables for data analysis. Used by:
    - Context building for AI agents
    - Multi-agent coordination
    - Analysis continuity between phases
    
    Example return:
    {
        'df': {
            'shape': (1000, 15),
            'columns': ['col1', 'col2', ...],
            'dtypes': {'col1': 'int64', 'col2': 'object', ...},
            'description': 'DataFrame(1000x15)'
        }
    }
    """
    dataframe_vars = {}
    core_imports = {'pd', 'np', 'stats', 'plt', 'sns'}
    
    for name, value in execution_namespace.items():
        if name.startswith('_') or name in core_imports:
            continue
            
        if isinstance(value, pd.DataFrame):
            try:
                dataframe_vars[name] = {
                    'shape': value.shape,
                    'columns': list(value.columns),
                    'dtypes': value.dtypes.to_dict(),
                    'description': f"DataFrame({value.shape[0]}x{value.shape[1]})"
                }
            except Exception:
                dataframe_vars[name] = {'description': "DataFrame(unknown shape)"}
    
    return dataframe_vars