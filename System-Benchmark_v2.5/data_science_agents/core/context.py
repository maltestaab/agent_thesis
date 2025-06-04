"""
data_science_agents/core/context.py - Simplified context for sharing state between agents
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass 
class AnalysisContext:
    """
    Simplified context object passed to tools and agents.
    
    Removed proxy methods - agents should call core functions directly.
    This reduces indirection and makes the code flow clearer.
    """
    
    # Basic analysis info
    file_name: str
    analysis_type: str  # "single_agent" or "multi_agent"
    start_time: float
    original_prompt: str = "" 
    
    # Execution state - updated by tools as work progresses
    completed_phases: Dict[str, str] = field(default_factory=dict)  # phase_name -> output
    agent_results: Dict[str, Any] = field(default_factory=dict)
    
    # Analytics tracker (will be attached by analysis systems)
    analytics: Optional[Any] = None
    
    def has_dataframes(self) -> bool:
        """
        Check if any dataframes are available in memory.
        Uses direct import to avoid circular dependencies.
        """
        from data_science_agents.core.execution import get_dataframe_variables
        return len(get_dataframe_variables()) > 0
    

def update_context_with_result(context: AnalysisContext, phase: str, result: Any):
    """Update context with results from a completed phase"""
    if phase not in context.completed_phases:
        context.completed_phases[phase] = str(result)
    context.agent_results[phase] = result