"""
data_science_agents/core/events.py - Shared events and helpers for streaming

This module provides:
1. Core event classes used throughout the streaming system
2. Helper functions to reduce duplication in event creation
3. Standardized event patterns for consistent UI updates

Purpose: Eliminates repetitive StreamingEvent creation code that was duplicated
across agent systems, making event creation consistent and easier to maintain.
"""
from dataclasses import dataclass
import time

@dataclass
class StreamingEvent:
    """
    Core event class for real-time UI updates in Streamlit.
    
    This represents a single update that gets sent to the UI during analysis.
    Different event types trigger different UI behaviors:
    - "text_delta": Incremental text updates (like ChatGPT typing)
    - "tool_call": When an agent starts using a tool (running code, etc.)
    - "tool_output": When tool execution completes
    - "analysis_complete": Final results ready
    - "analysis_error": Something went wrong
    - "analytics_complete": Final metrics ready
    
    Args:
        event_type: Category of event (determines UI handling)
        content: The actual message/data to display
        timestamp: When this event occurred (for ordering and timing)
        agent_name: Which agent generated this event (for multi-agent clarity)
    """
    event_type: str
    content: str
    timestamp: float
    agent_name: str = "Agent"


@dataclass
class AnalyticsEvent:
    """
    Event specifically for analytics/monitoring updates.
    
    Used internally for tracking agent performance, costs, and metrics.
    Separate from StreamingEvent to keep concerns separated.
    
    Args:
        event_type: Type of analytics event (agent_start, tool_call, cost_update, etc.)
        agent_name: Which agent this relates to
        timestamp: When the event occurred
        data: Additional metrics/data (flexible dict for different event types)
    """
    event_type: str
    agent_name: str
    timestamp: float
    data: dict = None


# =============================================================================
# EVENT CREATION HELPERS
# =============================================================================
# These functions eliminate repetitive event creation code that was scattered
# throughout the agent systems. Instead of manually creating StreamingEvent
# objects everywhere, we use these helpers for consistency.

def create_event(event_type: str, content: str, agent_name: str = "Agent") -> StreamingEvent:
    """
    Helper to create streaming events with automatic timestamp.
    
    This eliminates the repetitive pattern of:
        StreamingEvent(
            event_type="some_type",
            content="some content", 
            timestamp=time.time(),
            agent_name="some_agent"
        )
    
    Args:
        event_type: The type of event (determines UI handling)
        content: Message or data to display to user
        agent_name: Which agent is generating this event
        
    Returns:
        StreamingEvent ready to be yielded to the UI
        
    Example:
        # Instead of manual creation:
        yield StreamingEvent("tool_call", "🔧 Running analysis...", time.time(), "Data Scientist")
        
        # Use helper:
        yield create_event("tool_call", "🔧 Running analysis...", "Data Scientist")
    """
    return StreamingEvent(
        event_type=event_type,
        content=content,
        timestamp=time.time(),
        agent_name=agent_name
    )


def create_analysis_start_event(agent_mode: str) -> StreamingEvent:
    """
    Creates the standard analysis start event.
    
    This standardizes how we announce the beginning of analysis across
    both single-agent and multi-agent modes.
    
    Args:
        agent_mode: "single_agent" or "multi_agent" to customize the message
        
    Returns:
        StreamingEvent announcing analysis start
    """
    mode_name = "single agent" if agent_mode == "single_agent" else "multi-agent"
    return create_event(
        event_type="analysis_start",
        content=f"🚀 Starting {mode_name} analysis...",
        agent_name="System"
    )


def create_analysis_complete_event(final_output: str, duration: float, image_count: int) -> StreamingEvent:
    """
    Creates the standard analysis completion event.
    
    This provides a consistent format for announcing successful completion
    with key metrics that users care about.
    
    Args:
        final_output: The complete analysis results from the agent
        duration: How long the analysis took (in seconds)
        image_count: Number of visualizations created
        
    Returns:
        StreamingEvent with formatted completion message
    """
    metrics_text = f"✅ Analysis completed in {duration:.1f}s!"
    if image_count > 0:
        metrics_text += f"\n📊 Created {image_count} visualizations"
    
    complete_content = f"{metrics_text}\n\n**Final Results:**\n{final_output}"
    
    return create_event(
        event_type="analysis_complete",
        content=complete_content,
        agent_name="System"
    )


def create_tool_call_event(tool_description: str, agent_name: str = "Agent") -> StreamingEvent:
    """
    Creates a standardized tool call event.
    
    Tools are things like "execute Python code", "call specialist agent", etc.
    This provides consistent formatting for when agents start using tools.
    
    Args:
        tool_description: What the tool does (e.g., "executing code", "calling modeling agent")
        agent_name: Which agent is using the tool
        
    Returns:
        StreamingEvent announcing tool usage
    """
    return create_event(
        event_type="tool_call",
        content=f"🔧 {tool_description}...",
        agent_name=agent_name
    )


def create_tool_output_event(output_preview: str, agent_name: str = "Agent") -> StreamingEvent:
    """
    Creates a standardized tool output event.
    
    When tools finish executing, we show a preview of what they accomplished.
    This keeps the UI informative without overwhelming users with full output.
    
    Args:
        output_preview: Brief summary of what the tool accomplished
        agent_name: Which agent's tool just finished
        
    Returns:
        StreamingEvent showing tool completion
    """
    return create_event(
        event_type="tool_output",
        content=f"✅ {output_preview}",
        agent_name=agent_name
    )


def create_error_event(error_message: str, agent_name: str = "System") -> StreamingEvent:
    """
    Creates a standardized error event.
    
    When something goes wrong, we need to inform the user clearly.
    This provides consistent error formatting across the system.
    
    Args:
        error_message: Description of what went wrong
        agent_name: Which agent/system encountered the error
        
    Returns:
        StreamingEvent with formatted error message
    """
    return create_event(
        event_type="analysis_error",
        content=f"❌ {error_message}",
        agent_name=agent_name
    )


def create_cancellation_event() -> StreamingEvent:
    """
    Creates the standard cancellation event.
    
    When users click "Stop Analysis", this provides the consistent
    response they see.
    
    Returns:
        StreamingEvent announcing cancellation
    """
    return create_event(
        event_type="analysis_cancelled",
        content="🛑 Analysis cancelled by user",
        agent_name="System"
    )


def create_analytics_update_event(analytics_tracker) -> StreamingEvent:
    """
    Create an analytics update event with current metrics.
    
    This allows agent systems to periodically share their analytics
    with streamlit for live display updates without duplicating tracking.
    
    Args:
        analytics_tracker: AnalyticsTracker with current metrics
        
    Returns:
        StreamingEvent with structured analytics data for streamlit to parse
    """
    summary = analytics_tracker.get_summary()
    
    # Format as pipe-separated values that streamlit can easily parse
    analytics_content = f"ANALYTICS_UPDATE|{summary['total_duration']:.1f}|{summary['tool_calls']}|{summary['estimated_cost']:.4f}|{summary['images_created']}"
    
    return create_event(
        event_type="analytics_update",
        content=analytics_content,
        agent_name="System"
    )