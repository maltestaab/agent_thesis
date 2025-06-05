"""
data_science_agents/core/execution.py - Python Code Execution Engine

This module is the "hands" of the AI agents - it allows them to actually execute
Python code for data analysis rather than just talking about it. This transforms
AI agents from chatbots into functional data scientists.

Core Capabilities:
- Execute Python code written by AI agents
- Maintain variables and data between code executions
- Track created visualizations and files
- Provide context about available data and variables
- Handle errors gracefully with helpful feedback

Key Components:
- Code execution environment with persistent state
- Variable tracking and context awareness
- Image detection and cataloging
- Error handling and result formatting
- Integration with the agent tool system

Purpose: This is what makes the AI agents actually useful for data science.
Without this system, agents could only discuss data analysis. With it, they can:
- Load and manipulate real datasets
- Create professional visualizations  
- Build and evaluate machine learning models
- Generate actionable insights with actual numbers
- Persist results between analysis steps

The execution environment maintains state between tool calls, so agents can
build upon their previous work throughout an analysis session.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import sys
import io
import traceback
from agents import function_tool, RunContextWrapper
from data_science_agents.config.settings import IMAGES_DIR


# Suppress plotting warnings that can clutter agent output
# This improves the user experience by hiding technical matplotlib warnings
import warnings
warnings.filterwarnings('ignore')

# Ensure the Images directory exists for saving visualizations
# This guarantees agents can save plots without worrying about directory creation
os.makedirs(IMAGES_DIR, exist_ok=True)

# Global execution namespace - this is the "memory" where all variables live
# Pre-populated with common data science libraries to avoid import errors
execution_namespace = {
    'pd': pd,       # pandas for data manipulation
    'np': np,       # numpy for numerical operations  
    'stats': stats, # scipy.stats for statistical functions
    'plt': plt,     # matplotlib for plotting
    'sns': sns,     # seaborn for statistical visualizations
    'sm': sm       # statsmodels for statistical modeling
}

# Global tracking of images created during the current analysis session
# This helps distinguish new images from ones that existed before analysis
created_images = []


def reset_execution_state():
    """
    Reset the execution environment to a clean state for a new analysis.
    
    This function is called at the start of each analysis to ensure that
    variables and state from previous analyses don't interfere with new work.
    It's like clearing the workspace before starting a new project.
    
    What gets reset:
    - All user variables from previous analyses
    - Image tracking list
    - Execution namespace (except core libraries)
    
    What gets preserved:
    - Core library imports (pandas, numpy, etc.)
    - System configuration
    - The execution engine itself
    
    Purpose:
        - Prevents data contamination between analyses
        - Ensures consistent starting conditions
        - Avoids variable name conflicts
        - Provides clean slate for each user session
    """
    global execution_namespace, created_images
    
    # Clear all variables from the execution namespace
    execution_namespace.clear()
    
    # Restore essential library imports that agents expect to be available
    execution_namespace.update({
        'pd': pd,       # pandas - essential for data analysis
        'np': np,       # numpy - fundamental for numerical computing
        'stats': stats, # scipy.stats - needed for statistical analysis
        'plt': plt,     # matplotlib - required for plotting
        'sns': sns      # seaborn - popular for statistical visualization
    })
    
    # Clear the list of images created in this session
    created_images.clear()


@function_tool  # Decorator that makes this function available as a tool for AI agents
def execute_code(ctx: RunContextWrapper, code: str) -> str:
    """
    Execute Python code with context awareness and comprehensive result tracking.
    
    This is the main function that AI agents use to perform actual data analysis.
    It provides a safe, controlled environment for code execution with enhanced
    features that help agents work more effectively.
    
    Args:
        ctx (RunContextWrapper): Analysis context containing current state and history
        code (str): Python code string written by the AI agent
        
    Returns:
        str: Execution results, outputs, error messages, or success confirmation
        
    Key Features:
        1. Context Awareness: Tells agents what data/variables are already available
        2. Output Capture: Captures printed output and expression results  
        3. Image Tracking: Automatically detects new visualizations created
        4. Error Handling: Provides helpful error messages for debugging
        5. State Persistence: Variables remain available between executions
        6. Result Formatting: Returns clean, informative feedback to agents
        
    Workflow:
        1. Build context information for the agent
        2. Track existing images before execution
        3. Execute the code with output capture
        4. Detect any new images created
        5. Format and return comprehensive results
        
    This function is the bridge between AI reasoning and actual data science work.
    """

    try:
        # Build context information to help the agent work efficiently
        context_info = _build_enhanced_context_info(ctx)
        
        # Snapshot current images before execution to detect new ones
        images_before = _get_current_images()
        
        # Execute the code and capture all outputs
        output = _execute_with_capture(code)
        
        # Identify any new images created during execution
        new_images = _track_new_images(images_before)
        
        # Format and return comprehensive results
        return _format_result(output, new_images, context_info)
        
    except Exception as e:
        # Handle any errors that occur during execution
        return _format_error(e)


def _build_enhanced_context_info(ctx: RunContextWrapper) -> str:
    """
    Build comprehensive context information to help agents work efficiently.
    
    This function creates a "briefing" for AI agents about the current state
    of the analysis, including what data is available and what the user originally
    requested. This prevents agents from reloading data unnecessarily and helps
    them stay focused on the user's goals.
    
    Args:
        ctx (RunContextWrapper): Analysis context wrapper with current state
        
    Returns:
        str: Context information formatted as code comments
        
    Context Information Includes:
        - Original user request (so agents remember the goal)
        - Available dataframes (to avoid unnecessary reloading)
        - Other variables in memory (models, results, etc.)
        - Guidance on using existing data vs. loading new data
        
    Example Output:
        # ORIGINAL REQUEST: Analyze sales trends and predict next quarter
        # AVAILABLE DATAFRAMES: df, df_clean (use these instead of reloading)
        # OTHER VARIABLES: model_rf, accuracy_score, +2 more
    """
    # Return empty string if no context is available
    if not hasattr(ctx, 'context') or not ctx.context:
        return ""
    
    try:
        context_lines = []
        
        # Include original user request to keep agents focused on the goal
        if hasattr(ctx.context, 'original_prompt') and ctx.context.original_prompt:
            context_lines.append(f"# ORIGINAL REQUEST: {ctx.context.original_prompt}")
        
        # Get information about variables currently in memory
        available_vars = ctx.context.get_available_variables()
        if available_vars:
            # Separate dataframes from other variables for clarity
            dataframe_vars = [name for name, desc in available_vars.items() 
                            if 'DataFrame' in desc]
            other_vars = [name for name in available_vars.keys() if name not in dataframe_vars]
            
            # Highlight available dataframes to prevent unnecessary reloading
            if dataframe_vars:
                context_lines.append(f"# AVAILABLE DATAFRAMES: {', '.join(dataframe_vars)} (use these instead of reloading)")
            
            # List other variables, with truncation if there are too many
            if other_vars:
                if len(other_vars) <= 3:
                    context_lines.append(f"# OTHER VARIABLES: {', '.join(other_vars)}")
                else:
                    context_lines.append(f"# OTHER VARIABLES: {', '.join(other_vars[:3])}, +{len(other_vars)-3} more")
        
        # Return formatted context or empty string
        return '\n'.join(context_lines) + '\n' if context_lines else ""
    
    except Exception:
        # If context building fails, return empty string rather than crash
        return ""


def _get_current_images() -> set:
    """
    Get the current set of image files in the Images directory.
    
    This is used to detect new images created during code execution by
    comparing the directory contents before and after execution.
    
    Returns:
        set: Set of image filenames currently in the Images directory
        
    Purpose:
        - Enable detection of newly created visualizations
        - Support automatic image cataloging
        - Help agents understand what visual outputs they've produced
    """
    if not os.path.exists(IMAGES_DIR):
        return set()
    return set(os.listdir(IMAGES_DIR))


def _execute_with_capture(code: str) -> str:
    """
    Execute Python code and capture all printed output for return to the agent.
    
    This is the core function that actually runs the AI-generated Python code.
    It's designed to capture both explicit print statements and expression
    results, similar to how Jupyter notebooks work.
    
    Args:
        code (str): Python code string to execute
        
    Returns:
        str: Captured output from the code execution
        
    Key Features:
        - Output redirection to capture print statements
        - Expression evaluation for the last line of code
        - Safe execution in controlled namespace
        - Automatic result formatting
        
    How It Works:
        1. Redirect stdout to capture print statements
        2. Execute code in the global namespace
        3. Check if last line is an expression to evaluate
        4. Format results nicely for agent consumption
        5. Restore normal output handling
        
    Example:
        Code: "df.shape"
        Output: "Result: (1000, 15)"
        
        Code: "print('Hello'); df.head()"
        Output: "Hello\nResult: [DataFrame display]"
    """
    # Store original stdout and redirect to capture output
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    
    try:
        # Redirect stdout to capture print statements
        sys.stdout = captured_output
        
        # Execute the code in the global namespace where variables persist
        exec(code, execution_namespace)
        
        # Handle expression evaluation for the last line of code
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            
            # Check if the last line is an expression that should show output
            # This mimics Jupyter notebook behavior where expressions are displayed
            if last_line and not last_line.startswith('#') and (',' in last_line or last_line.isidentifier()):
                try:
                    # Evaluate the expression and format the result
                    result = eval(last_line, execution_namespace)
                    
                    # Handle tuple results specially for better readability
                    if isinstance(result, tuple):
                        print("Results:")
                        for i, item in enumerate(result):
                            print(f"Item {i+1}:")
                            print(item)
                            print("-" * 30)
                    else:
                        print(f"Result: {result}")
                        
                except Exception:
                    # If expression evaluation fails, just continue
                    # This handles cases where the last line isn't actually an expression
                    pass
        
        # Return all captured output
        return captured_output.getvalue()
        
    finally:
        # Always restore original stdout, even if execution fails
        sys.stdout = old_stdout


def _track_new_images(images_before: set) -> list:
    """
    Identify and register new images created during code execution.
    
    This function compares the Images directory before and after code execution
    to identify newly created visualizations, then registers them for display
    in the user interface.
    
    Args:
        images_before (set): Set of image filenames that existed before execution
        
    Returns:
        list: Sorted list of newly created image filenames
        
    Purpose:
        - Automatic detection of agent-created visualizations
        - Enable real-time display of new plots and charts
        - Track analysis outputs for final summary
        - Support image gallery functionality in UI
    """
    # Get current images and find new ones
    images_after = _get_current_images()
    new_images = sorted(images_after - images_before)
    
    # Register new images in global tracking list
    for img in new_images:
        img_path = os.path.join(IMAGES_DIR, img)
        if img_path not in created_images:
            created_images.append(img_path)
    
    return new_images


def _format_result(output: str, new_images: list, context_info: str) -> str:
    """
    Format execution results in a clean, informative way for agent consumption.
    
    This function takes the raw outputs from code execution and formats them
    into clear, actionable feedback that helps agents understand what happened
    and what they accomplished.
    
    Args:
        output (str): Captured output from code execution
        new_images (list): List of newly created image files
        context_info (str): Context information (currently not used in output)
        
    Returns:
        str: Formatted result string for the AI agent
        
    Formatting Rules:
        - Include actual output if any was generated
        - Report on newly created images
        - Provide clear success confirmation when no output is visible
        - Use informative messaging that helps agents understand results
        
    Purpose:
        - Clear communication of execution results
        - Help agents understand what they accomplished
        - Provide confirmation that code executed successfully
        - Support iterative development by agents
    """
    result_parts = []
    
    # Include actual output if any was generated
    if output.strip():
        result_parts.append(output.strip())
    
    # Report on newly created images
    if new_images:
        if len(new_images) == 1:
            result_parts.append(f"Created: {new_images[0]}")
        else:
            result_parts.append(f"Created {len(new_images)} images: {', '.join(new_images)}")
    
    # Provide appropriate feedback based on what happened
    if result_parts:
        return '\n'.join(result_parts)
    else:
        # Important: Let agents know code executed successfully even without visible output
        # This prevents agents from thinking execution failed and retrying unnecessarily
        return "Code executed successfully (no output to display)" 


def _format_error(error: Exception) -> str:
    """
    Format error messages to be helpful for AI agent debugging and learning.
    
    When code execution fails, this function transforms Python's sometimes
    cryptic error messages into clear, actionable feedback that helps AI agents
    understand what went wrong and how to fix it.
    
    Args:
        error (Exception): The exception that occurred during code execution
        
    Returns:
        str: Formatted error message with debugging information
        
    Error Information Includes:
        - Error type (what kind of error occurred)
        - Error message (specific description of the problem)
        - Relevant traceback (focused on the actual code problem)
        - Clean formatting that's easy for agents to parse
        
    Purpose:
        - Help agents understand and fix code problems
        - Provide actionable debugging information
        - Filter out irrelevant system traceback information
        - Enable agents to learn from mistakes and improve
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Extract relevant parts of the traceback
    tb_lines = traceback.format_exc().split('\n')
    relevant_tb = []
    
    # Filter out internal execution system lines, keep only useful information
    for line in tb_lines:
        # Skip the line that shows our internal exec() call
        if 'exec(code, execution_namespace)' in line:
            continue
        if line.strip():
            relevant_tb.append(line)
    
    # Build clear, actionable error message
    result = f"Error ({error_type}): {error_msg}"
    
    # Add traceback only if it provides useful information
    if len(relevant_tb) > 3:
        # Include the last 3 lines of traceback for context
        result += f"\n\nTraceback:\n" + '\n'.join(relevant_tb[-3:])
    
    return result


def get_created_images():
    """
    Get a list of all images created during the current analysis session.
    
    This function provides access to the complete list of visualizations
    created by agents during the analysis. It's used by the UI to display
    image galleries and by analytics to count visual outputs.
    
    Returns:
        list: Copy of the global created_images list
        
    Purpose:
        - Support image gallery display in the user interface
        - Enable analytics tracking of visual outputs
        - Provide agents with information about their created visualizations
        - Support final analysis summaries that reference all created images
        
    Note:
        Returns a copy to prevent external modification of the global list.
    """
    return created_images.copy()


def get_available_variables():
    """
    Get comprehensive information about all variables available in the execution environment.
    
    This function provides AI agents with detailed information about what data,
    models, and other variables are currently available in memory. This prevents
    unnecessary reloading of data and helps agents understand their working context.
    
    Returns:
        dict: Variable names mapped to their descriptive information
        
    Variable Descriptions Include:
        - DataFrames: Shape information (e.g., "DataFrame(1000x15)")
        - Arrays: Shape and type (e.g., "ndarray(100,)")  
        - Lists/Collections: Length (e.g., "list[25]")
        - Other Objects: Type name (e.g., "RandomForestRegressor")
        
    Purpose:
        - Inform agents about available data and models
        - Prevent unnecessary data reloading
        - Support context-aware analysis workflows
        - Enable smart variable reuse between analysis phases
        
    Example Return:
        {
            'df': 'DataFrame(1000x15)',
            'df_clean': 'DataFrame(985x15)', 
            'model_rf': 'RandomForestRegressor',
            'accuracy_score': 'float'
        }
    """
    user_vars = {}
    # Define core imports that should be excluded from variable listing
    core_imports = {'pd', 'np', 'stats', 'plt', 'sns'}
    
    for name, value in execution_namespace.items():
        # Skip private variables and core library imports
        if name.startswith('_') or name in core_imports:
            continue
            
        try:
            # Enhanced descriptions for different types of objects
            if isinstance(value, pd.DataFrame):
                # DataFrames get detailed shape information
                user_vars[name] = f"DataFrame({value.shape[0]}x{value.shape[1]})"
            elif hasattr(value, 'shape'):
                # Arrays and matrices show shape information
                user_vars[name] = f"{type(value).__name__}({value.shape})"
            elif hasattr(value, '__len__') and not isinstance(value, str):
                # Collections show length information
                user_vars[name] = f"{type(value).__name__}[{len(value)}]"
            else:
                # Other objects just show their type
                user_vars[name] = type(value).__name__
        except Exception:
            # If description generation fails, fall back to type name
            user_vars[name] = type(value).__name__
    
    return user_vars


def get_dataframe_variables():
    """
    Get detailed information about DataFrame variables specifically.
    
    DataFrames are the most important variables for data analysis, so this
    function provides comprehensive information about them specifically.
    This is used for context building and multi-agent coordination.
    
    Returns:
        dict: Detailed information about each available DataFrame
        
    DataFrame Information Includes:
        - Shape: Number of rows and columns
        - Columns: List of column names
        - Data Types: Type of each column
        - Description: Human-readable summary
        
    Purpose:
        - Support detailed DataFrame context for agents
        - Enable smart data reuse between analysis phases
        - Provide comprehensive data structure information
        - Support multi-agent coordination around shared datasets
        
    Example Return:
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
    # Define core imports to exclude
    core_imports = {'pd', 'np', 'stats', 'plt', 'sns'}
    
    for name, value in execution_namespace.items():
        # Skip private variables and core imports
        if name.startswith('_') or name in core_imports:
            continue
            
        # Process only DataFrame objects
        if isinstance(value, pd.DataFrame):
            try:
                # Collect comprehensive DataFrame information
                dataframe_vars[name] = {
                    'shape': value.shape,                          # (rows, columns)
                    'columns': list(value.columns),                # Column names
                    'dtypes': value.dtypes.to_dict(),             # Data types
                    'description': f"DataFrame({value.shape[0]}x{value.shape[1]})"  # Summary
                }
            except Exception:
                # If detailed information gathering fails, provide basic info
                dataframe_vars[name] = {'description': "DataFrame(unknown shape)"}
    
    return dataframe_vars