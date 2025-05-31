"""
data_science_agents/core/events.py - Shared events for streaming
"""
from dataclasses import dataclass
import time

@dataclass
class StreamingEvent:
    """Shared event for Streamlit updates"""
    event_type: str  # "text_delta", "tool_call", "tool_output", "agent_handoff", "message_complete", "analysis_complete"
    content: str
    timestamp: float
    agent_name: str = "Agent"

@dataclass
class AnalyticsEvent:
    """Analytics tracking event"""
    event_type: str  # "agent_start", "agent_end", "tool_call", "cost_update"
    agent_name: str
    timestamp: float
    data: dict = None