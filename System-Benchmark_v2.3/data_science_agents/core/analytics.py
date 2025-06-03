"""
data_science_agents/core/analytics.py - Analytics tracking for agent runs
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from data_science_agents.config.settings import DEFAULT_MODEL


@dataclass
class AgentTiming:
    """Track timing for individual agents"""
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tool_calls: int = 0
    
    def finish(self):
        """Mark agent as finished and calculate duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class AnalyticsTracker:
    """Track analytics for entire analysis run"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    
    # Agent tracking
    agent_timings: Dict[str, AgentTiming] = field(default_factory=dict)
    current_agent: Optional[str] = None
    
    # Tool tracking
    total_tool_calls: int = 0
    
    # Token tracking (approximate)
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    
    # Results
    phases_completed: List[str] = field(default_factory=list)
    images_created: List[str] = field(default_factory=list)
    
    def start_agent(self, agent_name: str):
        """Start tracking an agent"""
        if self.current_agent and self.current_agent in self.agent_timings:
            # Finish previous agent if still running
            self.agent_timings[self.current_agent].finish()
        
        self.current_agent = agent_name
        self.agent_timings[agent_name] = AgentTiming(
            agent_name=agent_name,
            start_time=time.time()
        )
    
    def finish_agent(self, agent_name: str):
        """Finish tracking an agent"""
        if agent_name in self.agent_timings:
            self.agent_timings[agent_name].finish()
            if agent_name not in self.phases_completed:
                self.phases_completed.append(agent_name)

    # Add these methods to the AnalyticsTracker class in analytics.py

    def estimate_tokens_from_content(self, content: str):
        """Estimate tokens from text content (rough: 4 chars = 1 token)"""
        estimated_tokens = len(content) / 4
        self.add_tokens(int(estimated_tokens))

    def update_cost_calculation(self):
        """Update cost calculation with current model rates (per million tokens)"""
        # Use gpt-4o-mini rates as default: $0.15 per 1M input tokens, $0.60 per 1M output tokens
        # Rough estimate: assume 70% input, 30% output
        input_tokens = int(self.estimated_tokens * 0.7)
        output_tokens = int(self.estimated_tokens * 0.3)
        
        self.estimated_cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

    # Update the add_tokens method to also update cost:
    def add_tokens(self, token_count: int):
        """Add to token estimate and update cost"""
        self.estimated_tokens += token_count
        self.update_cost_calculation()
        
    def add_tool_call(self, agent_name: str = None):
        """Track a tool call"""
        self.total_tool_calls += 1
        if agent_name and agent_name in self.agent_timings:
            self.agent_timings[agent_name].tool_calls += 1
    
    def add_image(self, image_path: str):
        """Track created image"""
        if image_path not in self.images_created:
            self.images_created.append(image_path)
    
    def finish(self):
        """Finish the entire analysis"""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        # Finish any running agent
        if self.current_agent and self.current_agent in self.agent_timings:
            self.agent_timings[self.current_agent].finish()
    
    def get_summary(self) -> Dict:
        """Get analytics summary"""
        agent_durations = {
            name: timing.duration or 0 
            for name, timing in self.agent_timings.items()
        }
        
        return {
            "total_duration": self.total_duration or (time.time() - self.start_time),
            "agent_durations": agent_durations,
            "tool_calls": self.total_tool_calls,
            "phases_completed": len(self.phases_completed),
            "estimated_cost": self.estimated_cost,
            "estimated_tokens": self.estimated_tokens,
            "images_created": len(self.images_created)
        }
    
