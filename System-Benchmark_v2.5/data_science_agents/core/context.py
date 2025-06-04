"""
data_science_agents/core/context.py - Analysis Context Management

This module provides the core context system that allows different parts of the 
analysis to share information and state. Think of it as a "shared notebook" that
all agents and tools can read from and write to during an analysis.

Key Components:
- AnalysisContext: Main data structure that travels with the analysis
- Context utilities: Helper functions for updating and managing context

Purpose: Enables coordination between different agents, tools, and analysis phases
by providing a centralized place to store and access shared information like:
- Original user request
- Analysis progress and results  
- Available data and variables
- Performance metrics

This is essential for multi-agent systems where specialists need to know what
previous agents have accomplished, and for maintaining continuity throughout
the analysis workflow.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass 
class AnalysisContext:
    """
    Central context object that carries shared state throughout an analysis.
    
    This acts as the "memory" of the analysis system, allowing different components
    to coordinate and build upon each other's work. It's like a shared workspace
    where all agents and tools can store and retrieve information.
    
    Key Design Principles:
    - Simple and direct: Agents access information directly rather than through proxy methods
    - Comprehensive: Contains all information needed for analysis coordination
    - Flexible: Can accommodate both single-agent and multi-agent workflows
    - Persistent: Maintains state throughout the entire analysis lifecycle
    
    Attributes:
        file_name (str): Name of the dataset file being analyzed
        analysis_type (str): Type of analysis ("single_agent" or "multi_agent")
        start_time (float): Unix timestamp when analysis began (for duration tracking)
        original_prompt (str): The user's original analysis request (so agents remember the goal)
        completed_phases (Dict[str, str]): Results from each completed analysis phase
        agent_results (Dict[str, Any]): Structured results from individual agents
        analytics (Optional[Any]): Performance tracking object (attached during analysis)
    
    Usage Example:
        # Create context for a new analysis
        context = AnalysisContext(
            file_name="sales_data.csv",
            analysis_type="multi_agent", 
            start_time=time.time(),
            original_prompt="Analyze sales trends and predict next quarter"
        )
        
        # Agents and tools can then read from and write to this context
        context.completed_phases["Data Understanding"] = "Found 3 key trends..."
        context.agent_results["modeling"] = {"accuracy": 0.85, "model": trained_model}
    """
    
    # === BASIC ANALYSIS INFORMATION ===
    # Core identifying information about the current analysis
    file_name: str              # Dataset being analyzed (e.g., "customer_data.csv")
    analysis_type: str          # Workflow type: "single_agent" or "multi_agent"  
    start_time: float           # When analysis began (for performance tracking)
    original_prompt: str = ""   # User's original request (preserved for agent reference)
    
    # === ANALYSIS PROGRESS TRACKING ===
    # These fields track what work has been completed and what results are available
    completed_phases: Dict[str, str] = field(default_factory=dict)  # phase_name -> summary_output
    agent_results: Dict[str, Any] = field(default_factory=dict)     # agent_name -> detailed_results
    
    # === PERFORMANCE MONITORING ===
    # Analytics tracker gets attached by the analysis systems for performance monitoring
    analytics: Optional[Any] = None  # AnalyticsTracker object (attached during setup)
    
    def has_dataframes(self) -> bool:
        """
        Check if any dataframes are currently available in the execution environment.
        
        This method helps agents determine whether they need to load data from files
        or can use dataframes that are already in memory from previous work.
        
        Returns:
            bool: True if dataframes are available in memory, False otherwise
            
        Usage:
            if context.has_dataframes():
                # Use existing dataframes (faster)
                agent_task = "Use the existing df to analyze patterns"
            else:
                # Need to load data first
                agent_task = "Load data from file and analyze patterns"
                
        Technical Note:
        Uses direct import to avoid circular dependencies between modules.
        The execution module manages the actual variable tracking.
        """
        from data_science_agents.core.execution import get_dataframe_variables
        return len(get_dataframe_variables()) > 0
    

def update_context_with_result(context: AnalysisContext, phase: str, result: Any):
    """
    Update the analysis context with results from a completed phase.
    
    This utility function provides a standardized way to record completed work
    in the context. It ensures that phase results are stored consistently
    and can be accessed by subsequent phases or agents.
    
    Args:
        context (AnalysisContext): The context object to update
        phase (str): Name of the completed phase (e.g., "Data Understanding")
        result (Any): The result from the phase (can be text, data, or complex objects)
        
    Purpose:
        - Prevents duplicate work by recording what's already been done
        - Allows later phases to build upon previous results
        - Maintains a complete record of the analysis journey
        - Provides consistent data structure for multi-agent coordination
        
    Example:
        # After a Data Understanding agent completes its work
        update_context_with_result(
            context, 
            "Data Understanding", 
            "Dataset has 1000 rows, 15 columns, 3% missing values in 'income' column"
        )
        
        # Later agents can then reference: context.completed_phases["Data Understanding"]
    """
    # Store summary text version for easy reference
    if phase not in context.completed_phases:
        context.completed_phases[phase] = str(result)
    
    # Store full result object for detailed access
    context.agent_results[phase] = result