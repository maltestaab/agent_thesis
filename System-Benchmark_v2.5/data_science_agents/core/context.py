"""
data_science_agents/core/context.py - Enhanced context for sharing state and variables between agents
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from data_science_agents.core.execution import get_available_variables, get_created_images, get_dataframe_variables


@dataclass 
class AnalysisContext:
    """
    Enhanced context object passed to tools and agents.
    
    This allows tools to access information about previous work
    and share dataframes/variables between agents without reloading files.
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
    
    def get_available_variables(self) -> Dict[str, str]:
        """Get currently available variables in execution namespace"""
        return get_available_variables()
    
    def get_dataframe_variables(self) -> Dict[str, Dict]:
        """Get detailed information about available dataframes"""
        return get_dataframe_variables()
    
    def get_created_images(self) -> List[str]:
        """Get list of created images"""
        return get_created_images()
    
    def has_dataframes(self) -> bool:
        """Check if any dataframes are available in memory"""
        return len(self.get_dataframe_variables()) > 0
    

def update_context_with_result(context: AnalysisContext, phase: str, result: Any):
    """Update context with results from a completed phase"""
    if phase not in context.completed_phases:
        context.completed_phases[phase] = str(result)
    context.agent_results[phase] = result