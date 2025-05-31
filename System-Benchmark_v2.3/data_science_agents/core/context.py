"""
data_science_agents/core/context.py - Simple context for sharing state between agents and tools
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any
from data_science_agents.core.execution import get_available_variables, get_created_images


@dataclass 
class AnalysisContext:
    """
    Simple context object passed to tools and agents.
    
    This allows tools to access information about previous work
    without re-executing or re-analyzing what's already been done.
    """
    
    # Basic analysis info
    file_name: str
    analysis_type: str  # "single_agent" or "multi_agent"
    start_time: float
    
    # Execution state - updated by tools as work progresses
    completed_phases: List[str] = field(default_factory=list)
    agent_results: Dict[str, Any] = field(default_factory=dict)
    
    def get_available_variables(self) -> Dict[str, str]:
        """Get currently available variables in execution namespace"""
        return get_available_variables()
    
    def get_created_images(self) -> List[str]:
        """Get list of created images"""
        return get_created_images()

def update_context_with_result(context: AnalysisContext, phase: str, result: Any):
    """Update context with results from a completed phase"""
    context.completed_phases.append(phase)
    context.agent_results[phase] = result




"""
data_science_agents/core/events.py - Shared events for streaming
"""

