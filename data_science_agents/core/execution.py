"""
Core execution functionality for data science agents
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Any, Dict, List

class ExecutionState:
    def __init__(self):
        self.namespace: Dict[str, Any] = {
            'pd': pd,
            'np': np,
            'stats': stats
        }
        self.results: List[str] = []
        self.images: List[str] = []
        self.analysis_history: List[Dict[str, Any]] = []
    
    def store_result(self, result: Any) -> None:
        """Store a result in the analysis history"""
        self.results.append(str(result))
    
    def add_to_namespace(self, name: str, value: Any) -> None:
        """Add a variable to the namespace"""
        self.namespace[name] = value
    
    def get_from_namespace(self, name: str) -> Any:
        """Get a variable from the namespace"""
        return self.namespace.get(name)
    
    def store_analysis(self, phase: str, result: Any) -> None:
        """Store results from an analysis phase"""
        self.analysis_history.append({
            'phase': phase,
            'result': result,
            'namespace': {k: v for k, v in self.namespace.items() if not k.startswith('_')}
        })
    
    def get_previous_analysis(self, phase: str) -> Any:
        """Get results from a specific previous phase"""
        for entry in reversed(self.analysis_history):
            if entry['phase'] == phase:
                return entry['result']
        return None
    
    def reset(self) -> None:
        """Reset the execution state"""
        self.__init__()

# Global state instance
_state = ExecutionState()

def reset_execution_state():
    """Reset the execution state"""
    _state.reset()

def execute_code(code: str) -> str:
    """Execute Python code and return the result"""
    try:
        # Execute the code in the state's namespace
        exec(code, _state.namespace)
        
        # Return any stored results
        if _state.results:
            return '\n'.join(_state.results)
        return "Code executed successfully"
        
    except Exception as e:
        return f"Error executing code: {str(e)}"

def get_created_images() -> List[str]:
    """Get any images created during execution"""
    return _state.images

# Export the state object for direct access
state = _state