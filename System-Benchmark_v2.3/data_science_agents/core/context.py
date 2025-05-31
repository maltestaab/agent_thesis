"""
data_science_agents/core/context.py - Simple context for sharing state between agents and tools
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any
from agents import RunContextWrapper
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
    
    def has_data_loaded(self) -> bool:
        """Check if data has already been loaded"""
        variables = self.get_available_variables()
        # Common variable names for loaded datasets
        data_vars = ['df', 'data', 'dataset']
        return any(var in variables for var in data_vars)
    
    def get_previous_results_summary(self) -> str:
        """Get a summary of what's been accomplished so far"""
        if not self.agent_results:
            return "No previous analysis results available."
        
        summary_parts = []
        for phase, result in self.agent_results.items():
            if hasattr(result, 'summary'):
                summary_parts.append(f"**{phase}**: {result.summary}")
            else:
                summary_parts.append(f"**{phase}**: Completed")
        
        return "\n".join(summary_parts)


def update_context_with_result(context: AnalysisContext, phase: str, result: Any):
    """Update context with results from a completed phase"""
    context.completed_phases.append(phase)
    context.agent_results[phase] = result