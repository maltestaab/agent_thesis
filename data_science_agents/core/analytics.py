"""
Enhanced analytics system for tracking agent performance
"""
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from agents import AgentHooks

@dataclass
class ToolMetrics:
    """Metrics for tool usage"""
    total_calls: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    
    @classmethod
    def from_usage_data(cls, tool_usage: List[Dict[str, Any]]) -> Dict[str, 'ToolMetrics']:
        """Create tool metrics from raw usage data"""
        metrics_by_tool = {}
        for usage in tool_usage:
            tool_name = usage['tool']
            if tool_name not in metrics_by_tool:
                metrics_by_tool[tool_name] = cls()
            
            metrics = metrics_by_tool[tool_name]
            metrics.total_calls += 1
            if 'duration' in usage:
                metrics.total_duration += usage['duration']
            if 'error' in usage:
                metrics.error_count += 1
        
        # Calculate averages and rates
        for metrics in metrics_by_tool.values():
            if metrics.total_calls > 0:
                metrics.average_duration = metrics.total_duration / metrics.total_calls
                metrics.error_rate = metrics.error_count / metrics.total_calls
        
        return metrics_by_tool

@dataclass
class AgentMetrics:
    """Metrics for agent performance"""
    total_time: float = 0.0
    thinking_time: float = 0.0  # Time not spent in tools
    tool_time: float = 0.0
    tool_calls: int = 0
    error_count: int = 0
    
    @classmethod
    def from_usage_data(cls, agent_name: str, tool_usage: List[Dict[str, Any]], memory_usage: List[Dict[str, Any]]) -> 'AgentMetrics':
        """Create agent metrics from raw usage data"""
        metrics = cls()
        
        # Calculate agent runtime from memory usage
        agent_starts = [m for m in memory_usage if m['agent'] == agent_name and m['event'] == 'start']
        agent_ends = [m for m in memory_usage if m['agent'] == agent_name and m['event'] == 'end']
        if agent_starts and agent_ends:
            metrics.total_time = agent_ends[-1]['timestamp'] - agent_starts[0]['timestamp']
        
        # Calculate tool usage time and counts
        agent_tool_usage = [t for t in tool_usage if t['agent'] == agent_name]
        metrics.tool_calls = len(agent_tool_usage)
        metrics.tool_time = sum(t.get('duration', 0) for t in agent_tool_usage)
        metrics.error_count = sum(1 for t in agent_tool_usage if 'error' in t)
        
        # Thinking time is total time minus tool time
        metrics.thinking_time = metrics.total_time - metrics.tool_time
        
        return metrics

@dataclass
class AnalyticsData:
    """Container for analytics data"""
    # Basic metrics
    duration: float = 0.0
    steps_count: int = 0
    agent_calls: int = 0
    
    # Raw data
    tool_usage: List[Dict[str, Any]] = field(default_factory=list)
    memory_usage: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    created_images: List[str] = field(default_factory=list)
    
    # Computed metrics
    tool_metrics: Dict[str, ToolMetrics] = field(default_factory=dict)
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    
    def compute_metrics(self):
        """Compute all metrics from raw data"""
        # Compute tool metrics
        self.tool_metrics = ToolMetrics.from_usage_data(self.tool_usage)
        
        # Compute agent metrics
        agent_names = {usage['agent'] for usage in self.tool_usage}
        self.agent_metrics = {
            name: AgentMetrics.from_usage_data(name, self.tool_usage, self.memory_usage)
            for name in agent_names
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for comparison"""
        return {
            'total_duration': self.duration,
            'total_steps': self.steps_count,
            'total_agents': self.agent_calls,
            'total_tool_calls': len(self.tool_usage),
            'total_errors': len(self.errors),
            'average_tool_duration': statistics.mean([t.get('duration', 0) for t in self.tool_usage if 'duration' in t]) if self.tool_usage else 0,
            'total_images': len(self.created_images),
            'error_rate': len(self.errors) / len(self.tool_usage) if self.tool_usage else 0
        }

class AnalyticsHooks(AgentHooks):
    """Hooks for gathering detailed analytics during agent runs"""
    
    def __init__(self):
        self.start_time = time.time()
        self.tool_usage = []
        self.memory_usage = []
        self.errors = []
        self.current_tool_start = None
    
    async def before_agent_run(self, agent: Agent, context: Any):
        """Track when an agent starts"""
        self.memory_usage.append({
            'timestamp': time.time(),
            'agent': agent.name,
            'event': 'start'
        })
    
    async def after_agent_run(self, agent: Agent, context: Any, result: Any):
        """Track when an agent completes"""
        self.memory_usage.append({
            'timestamp': time.time(),
            'agent': agent.name,
            'event': 'end'
        })
    
    async def on_tool_call(self, agent: Agent, context: Any, tool_name: str):
        """Track tool usage start"""
        self.current_tool_start = time.time()
        self.tool_usage.append({
            'agent': agent.name,
            'tool': tool_name,
            'start_time': self.current_tool_start
        })
    
    async def on_tool_end(self, agent: Agent, context: Any, tool_name: str):
        """Track tool usage completion"""
        if self.current_tool_start:
            end_time = time.time()
            # Update the last matching tool usage record
            for record in reversed(self.tool_usage):
                if (record['agent'] == agent.name and 
                    record['tool'] == tool_name and 
                    'end_time' not in record):
                    record['end_time'] = end_time
                    record['duration'] = end_time - record['start_time']
                    break
            self.current_tool_start = None
    
    async def on_tool_error(self, agent: Agent, context: Any, error: Exception):
        """Track tool errors"""
        self.errors.append({
            'timestamp': time.time(),
            'agent': agent.name,
            'error': str(error)
        })
        if self.current_tool_start:
            self.tool_usage[-1]['error'] = str(error)
    
    def get_analytics_data(self, steps_count: int, agent_calls: int) -> AnalyticsData:
        """Compile all analytics data"""
        analytics = AnalyticsData(
            duration=time.time() - self.start_time,
            steps_count=steps_count,
            agent_calls=agent_calls,
            tool_usage=self.tool_usage,
            memory_usage=self.memory_usage,
            errors=self.errors
        )
        analytics.compute_metrics()
        return analytics

# Analytics instances for different agent systems
single_analytics = AnalyticsHooks()
multi_analytics = AnalyticsHooks()