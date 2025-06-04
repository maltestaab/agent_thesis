"""
data_science_agents/core/events.py - Streaming Event System

This module provides the event system that enables real-time communication between
the analysis engine and the user interface. It's like a "news feed" that reports
what's happening during analysis so users can watch progress in real-time.

Key Components:
- Event Classes: Data structures that represent different types of updates
- Event Helpers: Functions that create standardized events consistently
- Analytics Events: Special events for sharing performance metrics

Purpose: Enables live streaming of analysis progress to create an interactive,
transparent user experience. Instead of waiting for analysis to complete,
users see each step as it happens - like watching code execute in real-time.

Event Types:
- Text streaming: See agent reasoning as it's generated (like ChatGPT)
- Tool activity: Watch when agents execute code or use tools
- Progress updates: Track which analysis phases are starting/completing
- Analytics: Monitor performance, cost, and resource usage live
- Status changes: Handle completion, errors, and cancellation

This system transforms a "black box" analysis into a transparent, engaging
process where users understand exactly what's happening and why.
"""
from dataclasses import dataclass
import time

@dataclass
class StreamingEvent:
    """
    Core event class that represents a single real-time update to the user interface.
    
    Each StreamingEvent is like a "message" sent from the analysis engine to the
    web interface, telling it to update the display with new information. This
    creates the live streaming effect users see during analysis.
    
    Event Categories and Their Purpose:
    - "text_delta": Incremental text updates (agent reasoning appears character by character)
    - "tool_call": Agent is about to execute code or use a tool
    - "tool_output": Tool execution completed, showing results
    - "specialist_start": A specialist agent begins working (multi-agent mode)
    - "specialist_complete": A specialist agent finishes its work
    - "analysis_complete": Entire analysis finished successfully
    - "analysis_error": Something went wrong during analysis
    - "analytics_update": Live performance metrics update
    - "analytics_complete": Final performance summary
    
    Attributes:
        event_type (str): Category of event (determines how UI handles it)
        content (str): The actual message or data to display to the user
        timestamp (float): When this event occurred (for ordering and timing)
        agent_name (str): Which agent generated this event (helps users track who's working)
        
    Example Usage:
        # Create an event when agent starts thinking
        event = StreamingEvent(
            event_type="text_delta",
            content="I need to analyze the correlation between...",
            timestamp=time.time(),
            agent_name="Data Science Agent"
        )
        
        # Send to UI (typically done via yield in streaming functions)
        yield event
    """
    event_type: str             # What kind of update this represents
    content: str                # The actual message/data for the user  
    timestamp: float            # When this happened (Unix timestamp)
    agent_name: str = "Agent"   # Which agent is responsible for this event


@dataclass
class AnalyticsEvent:
    """
    Specialized event for internal analytics and performance monitoring.
    
    While StreamingEvent handles user-facing updates, AnalyticsEvent manages
    internal performance tracking. This separation keeps user experience
    events distinct from system monitoring data.
    
    Used for tracking:
    - Agent start/stop times
    - Tool usage statistics  
    - Cost accumulation
    - Resource consumption
    - Performance bottlenecks
    
    Attributes:
        event_type (str): Type of analytics event (agent_start, tool_call, etc.)
        agent_name (str): Which agent/component this relates to
        timestamp (float): When the event occurred
        data (dict): Additional metrics or data specific to this event type
        
    Note: This is primarily for internal system monitoring rather than user display.
    """
    event_type: str             # Category of analytics event
    agent_name: str             # Which agent/component is involved
    timestamp: float            # When this measurement was taken
    data: dict = None           # Additional metrics (flexible structure)


# =============================================================================
# EVENT CREATION HELPERS
# =============================================================================
# These functions provide standardized ways to create events, eliminating
# repetitive code and ensuring consistent formatting across the system.

def create_event(event_type: str, content: str, agent_name: str = "Agent") -> StreamingEvent:
    """
    Helper function to create streaming events with automatic timestamp.
    
    This eliminates the repetitive work of manually creating StreamingEvent objects
    and ensures all events have consistent structure and timing information.
    
    Args:
        event_type (str): The category of event (determines UI behavior)
        content (str): Message or data to display to the user
        agent_name (str): Which agent is generating this event
        
    Returns:
        StreamingEvent: Properly formatted event ready to send to UI
        
    Benefits:
        - Automatic timestamp generation
        - Consistent event structure
        - Reduced code duplication
        - Centralized event formatting
        
    Example:
        # Instead of manual event creation:
        event = StreamingEvent(
            event_type="tool_call",
            content="🔧 Running analysis...", 
            timestamp=time.time(),
            agent_name="Data Scientist"
        )
        
        # Use the helper:
        event = create_event("tool_call", "🔧 Running analysis...", "Data Scientist")
    """
    return StreamingEvent(
        event_type=event_type,
        content=content,
        timestamp=time.time(),  # Automatic timestamp
        agent_name=agent_name
    )


def create_analysis_start_event(agent_mode: str) -> StreamingEvent:
    """
    Create the standard event that announces the beginning of analysis.
    
    This provides consistent messaging when analysis begins, customized for
    the specific analysis mode being used.
    
    Args:
        agent_mode (str): "single_agent" or "multi_agent" for customized messaging
        
    Returns:
        StreamingEvent: Start announcement event
        
    Purpose:
        - Consistent start messaging across different analysis modes
        - Clear indication to users that analysis is beginning
        - Sets expectations for the type of workflow being used
    """
    # Customize message based on analysis mode
    mode_name = "single agent" if agent_mode == "single_agent" else "multi-agent"
    return create_event(
        event_type="analysis_start",
        content=f"🚀 Starting {mode_name} analysis...",
        agent_name="System"
    )


def create_analysis_complete_event(final_output: str, duration: float, image_count: int) -> StreamingEvent:
    """
    Create the standard completion event with key success metrics.
    
    This provides a consistent format for announcing successful analysis
    completion, including the metrics users care most about.
    
    Args:
        final_output (str): The complete analysis results from the agent
        duration (float): How long the analysis took (in seconds)
        image_count (int): Number of visualizations created
        
    Returns:
        StreamingEvent: Completion announcement with results
        
    Purpose:
        - Celebrate successful completion
        - Provide key performance metrics
        - Include the final analysis results
        - Highlight visual outputs created
    """
    # Build success message with metrics
    metrics_text = f"✅ Analysis completed in {duration:.1f}s!"
    if image_count > 0:
        metrics_text += f"\n📊 Created {image_count} visualizations"
    
    # Combine metrics with results
    complete_content = f"{metrics_text}\n\n**Final Results:**\n{final_output}"
    
    return create_event(
        event_type="analysis_complete",
        content=complete_content,
        agent_name="System"
    )


def create_tool_call_event(tool_description: str, agent_name: str = "Agent") -> StreamingEvent:
    """
    Create a standardized event for when agents start using tools.
    
    Tools are external capabilities like executing Python code, calling other
    agents, or accessing databases. This event lets users know when agents
    are taking action rather than just thinking.
    
    Args:
        tool_description (str): What the tool does (e.g., "executing code")
        agent_name (str): Which agent is using the tool
        
    Returns:
        StreamingEvent: Tool usage announcement
        
    Purpose:
        - Transparency into when agents take action
        - Clear indication of tool usage vs. reasoning
        - Consistent formatting for all tool activities
    """
    return create_event(
        event_type="tool_call",
        content=f"🔧 {tool_description}...",
        agent_name=agent_name
    )


def create_tool_output_event(output_preview: str, agent_name: str = "Agent") -> StreamingEvent:
    """
    Create a standardized event for when tool execution completes.
    
    After tools finish running, this event shows users a preview of what
    was accomplished, providing closure and context for the tool usage.
    
    Args:
        output_preview (str): Brief summary of what the tool accomplished
        agent_name (str): Which agent's tool just finished
        
    Returns:
        StreamingEvent: Tool completion notification
        
    Purpose:
        - Show results of tool usage
        - Provide closure for tool activities
        - Keep users informed without overwhelming detail
    """
    return create_event(
        event_type="tool_output",
        content=f"✅ {output_preview}",
        agent_name=agent_name
    )


def create_error_event(error_message: str, agent_name: str = "System") -> StreamingEvent:
    """
    Create a standardized event for error conditions.
    
    When something goes wrong during analysis, this provides consistent
    error formatting and clear communication to users about what happened.
    
    Args:
        error_message (str): Description of what went wrong
        agent_name (str): Which agent/system encountered the error
        
    Returns:
        StreamingEvent: Error notification
        
    Purpose:
        - Clear error communication
        - Consistent error formatting
        - Helpful for debugging and user understanding
    """
    return create_event(
        event_type="analysis_error",
        content=f"❌ {error_message}",
        agent_name=agent_name
    )


def create_cancellation_event() -> StreamingEvent:
    """
    Create the standard event for user-initiated cancellation.
    
    When users click "Stop Analysis", this provides the consistent response
    they see, confirming that their cancellation request was received.
    
    Returns:
        StreamingEvent: Cancellation confirmation
        
    Purpose:
        - Confirm cancellation request was received
        - Provide closure when analysis is stopped
        - Consistent messaging for user-initiated stops
    """
    return create_event(
        event_type="analysis_cancelled",
        content="🛑 Analysis cancelled by user",
        agent_name="System"
    )


def create_analytics_update_event(analytics_tracker) -> StreamingEvent:
    """
    Create an analytics update event for live performance monitoring.
    
    This allows the analysis systems to periodically share their performance
    metrics with the user interface for real-time display of costs, timing,
    and progress indicators.
    
    Args:
        analytics_tracker: AnalyticsTracker object with current metrics
        
    Returns:
        StreamingEvent: Analytics data formatted for UI parsing
        
    Purpose:
        - Share performance metrics in real-time
        - Enable live cost and timing displays
        - Maintain separation between analytics tracking and UI display
        - Provide structured data that UI can easily parse
        
    Data Format:
        The content is formatted as pipe-separated values for easy parsing:
        "ANALYTICS_UPDATE|duration|tool_calls|cost|images_created"
        
    Example:
        Content might be: "ANALYTICS_UPDATE|45.2|8|0.0142|3"
        Meaning: 45.2 seconds, 8 tool calls, $0.0142 cost, 3 images created
    """
    # Get current metrics from the analytics tracker
    summary = analytics_tracker.get_summary()
    
    # Format as structured data that the UI can easily parse
    # Using pipe separation for reliability and simplicity
    analytics_content = f"ANALYTICS_UPDATE|{summary['total_duration']:.1f}|{summary['tool_calls']}|{summary['estimated_cost']:.4f}|{summary['images_created']}"
    
    return create_event(
        event_type="analytics_update",
        content=analytics_content,
        agent_name="System"
    )