"""
data_science_agents/core/analytics.py - Comprehensive Analytics Tracking System

This module provides the core analytics infrastructure for monitoring and tracking
all aspects of data science analysis performance:

Key Features:
- Real-time tracking of agent timing, tool usage, and token consumption
- Model-specific cost calculation using current pricing from OpenAI
- Support for both single-agent and multi-agent analysis modes
- Integration with the context system for state sharing across components
- Helper functions to eliminate code duplication across agent systems

Purpose:
This serves as the single source of truth for all analysis metrics, enabling
accurate cost tracking, performance monitoring, and real-time UI updates.
The analytics system integrates seamlessly with both analysis modes while
maintaining clear separation of concerns.
"""

import time
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from contextlib import contextmanager

from data_science_agents.config.settings import get_model_cost, DEFAULT_MODEL

if TYPE_CHECKING:
    from data_science_agents.core.context import AnalysisContext


@dataclass
class AgentTiming:
    """
    Performance metrics tracker for individual agents.
    
    In single-agent mode, this tracks the one comprehensive agent.
    In multi-agent mode, this provides detailed timing for each specialist,
    enabling performance analysis and system optimization.
    
    Attributes:
        agent_name (str): Human-readable agent identifier
        start_time (float): Unix timestamp when agent began work
        end_time (float): Unix timestamp when agent finished (None if running)
        duration (float): Calculated runtime in seconds (None until finished)
        tool_calls (int): Number of tools this agent used during execution
    """
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tool_calls: int = 0
    
    def finish(self):
        """
        Mark this agent as completed and calculate final duration.
        
        This method is called when an agent completes its work, either
        successfully or due to an error. It ensures accurate timing data
        for performance analysis.
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class AnalyticsTracker:
    """
    Comprehensive analytics tracking for data science analysis sessions.
    
    This class serves as the central hub for all performance metrics during
    an analysis. It tracks everything from basic timing to detailed cost
    estimation and provides real-time data for UI updates.
    
    Key Responsibilities:
    1. Track timing for overall analysis and individual agents
    2. Monitor tool usage (code execution, agent calls, etc.)
    3. Estimate token consumption and calculate costs using model-specific pricing
    4. Provide real-time summaries for UI display and monitoring
    5. Support both single-agent and multi-agent analysis modes
    
    Design Philosophy:
    - Single source of truth for all analysis metrics
    - Model-aware cost calculation for accurate budgeting
    - Real-time updates for immediate user feedback
    - Clean separation from execution concerns (images tracked in execution.py)
    """
    
    # === TIMING TRACKING ===
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    
    # === AGENT PERFORMANCE TRACKING ===
    agent_timings: Dict[str, AgentTiming] = field(default_factory=dict)
    current_agent: Optional[str] = None
    
    # === TOOL USAGE TRACKING ===
    total_tool_calls: int = 0
    
    # === TOKEN AND COST TRACKING ===
    # Separated input/output tokens for accurate cost calculation
    # OpenAI charges different rates for input vs output tokens
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost: float = 0.0
    
    # === MODEL CONFIGURATION ===
    # Stores which AI model is being used for accurate cost calculation
    model_name: str = DEFAULT_MODEL
    
    # === PHASE COMPLETION TRACKING ===
    # Tracks which CRISP-DM phases have been completed
    phases_completed: List[str] = field(default_factory=list)
    
    def start_agent(self, agent_name: str):
        """
        Begin tracking a new agent's performance.
        
        This method is called when an agent starts work. In single-agent mode,
        there's typically just one call. In multi-agent mode, this is called
        for each specialist agent as they take control of their phase.
        
        Args:
            agent_name (str): Human-readable agent name for tracking
        """
        # Finish the previous agent if one was running
        if self.current_agent and self.current_agent in self.agent_timings:
            self.agent_timings[self.current_agent].finish()
        
        # Start tracking the new agent
        self.current_agent = agent_name
        self.agent_timings[agent_name] = AgentTiming(
            agent_name=agent_name,
            start_time=time.time()
        )
    
    def finish_agent(self, agent_name: str):
        """
        Mark an agent as completed and record its final metrics.
        
        This method should be called when an agent completes its work,
        regardless of whether it finished successfully or encountered an error.
        
        Args:
            agent_name (str): Name of the agent that just finished
        """
        if agent_name in self.agent_timings:
            self.agent_timings[agent_name].finish()
            # Track completion for phase counting
            if agent_name not in self.phases_completed:
                self.phases_completed.append(agent_name)

    def estimate_tokens_from_content(self, content: str):
        """
        Estimate token usage from text content during streaming.
        
        This method is called during real-time streaming when we receive text
        from the AI model. We assume this is output content (agent responses)
        rather than input content.
        
        Token Estimation Rule: Approximately 4 characters = 1 token
        This is OpenAI's commonly cited rule of thumb for English text.
        
        Args:
            content (str): Text content from AI model response
        """
        estimated_tokens = len(content) / 4
        self.add_output_tokens(int(estimated_tokens))

    def add_input_tokens(self, token_count: int):
        """
        Add input tokens and update cost calculation.
        
        Input tokens represent what we send TO the AI model, including
        prompts, context, data descriptions, and previous conversation history.
        These typically cost less than output tokens.
        
        Args:
            token_count (int): Number of input tokens to add
        """
        self.estimated_input_tokens += token_count
        self._update_cost_calculation()
        
    def add_output_tokens(self, token_count: int):
        """
        Add output tokens and update cost calculation.
        
        Output tokens represent what we get FROM the AI model, including
        analysis, code, insights, and reasoning. These typically cost more
        than input tokens in most pricing models.
        
        Args:
            token_count (int): Number of output tokens to add
        """
        self.estimated_output_tokens += token_count
        self._update_cost_calculation()

    def _update_cost_calculation(self):
        """
        Update cost calculation using model-specific rates from settings.
        
        This method is called automatically whenever tokens are added. It ensures
        that cost estimates are always current and use the correct pricing
        for whichever AI model the user selected.
        
        Why Model-Specific Pricing Matters:
        - gpt-4o-mini: $0.15 input, $0.60 output (per 1M tokens)
        - gpt-4o: $3.00 input, $10.00 output (per 1M tokens)
        - gpt-4: $30.00 input, $60.00 output (per 1M tokens)
        
        The cost difference between models can be 20x or more!
        """
        self.estimated_cost = get_model_cost(
            self.model_name, 
            self.estimated_input_tokens, 
            self.estimated_output_tokens
        )
        
    def add_tool_call(self, agent_name: str = None):
        """
        Record that an agent used a tool.
        
        Tools include various capabilities like executing Python code,
        calling other agents, searching databases, etc. This helps track
        how much computational work agents are doing.
        
        Args:
            agent_name (str): Which agent used the tool (for per-agent tracking)
        """
        self.total_tool_calls += 1
        if agent_name and agent_name in self.agent_timings:
            self.agent_timings[agent_name].tool_calls += 1
    
    def set_model(self, model_name: str):
        """
        Set the AI model being used and recalculate costs.
        
        This method should be called when the analysis starts to ensure cost
        calculations use the correct pricing for the selected model. It also
        handles cases where the model might change during analysis.
        
        Args:
            model_name (str): Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
        """
        self.model_name = model_name
        self._update_cost_calculation()  # Recalculate with new model rates
    
    def finish(self):
        """
        Mark the entire analysis as completed and finalize all metrics.
        
        This method should be called when the analysis is completely done,
        regardless of success or failure. It ensures all timing data is
        properly calculated and any running agents are marked as finished.
        """
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        # Finish any agent that's still running
        if self.current_agent and self.current_agent in self.agent_timings:
            self.agent_timings[self.current_agent].finish()
    
    def get_summary(self) -> Dict:
        """
        Get complete analytics summary for display and storage.
        
        This method provides all key metrics in a standardized format that
        can be used by the UI, stored in session state, or logged for
        system monitoring and analysis.
        
        Returns:
            Dict: Complete analytics data including:
                - total_duration: Overall analysis time in seconds
                - agent_durations: Per-agent timing breakdown
                - tool_calls: Total number of tool uses across all agents
                - phases_completed: Number of CRISP-DM phases finished
                - estimated_cost: Total estimated cost in USD
                - token counts: Detailed input/output token usage
                - images_created: Number of visualizations (from execution.py)
        """
        # Import here to avoid circular import issues
        from data_science_agents.core.execution import get_created_images
        
        # Calculate per-agent durations for performance analysis
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
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "images_created": len(get_created_images())  # Single source from execution.py
        }


# =============================================================================
# ANALYTICS SETUP HELPERS
# =============================================================================

def setup_analytics(context: "AnalysisContext", agent_name: str, model_name: str = DEFAULT_MODEL) -> AnalyticsTracker:
    """
    Initialize analytics tracking and attach it to the analysis context.
    
    This helper function provides a standardized way to set up analytics
    for both single-agent and multi-agent systems. It eliminates code
    duplication and ensures consistent configuration across all analysis modes.
    
    Setup Process:
    1. Create a new AnalyticsTracker instance
    2. Configure it with the selected AI model for accurate cost calculation
    3. Attach it to the analysis context for sharing between components
    4. Start tracking the primary agent
    
    Args:
        context (AnalysisContext): Context object that will carry the analytics
        agent_name (str): Name of the main agent starting the analysis
        model_name (str): AI model being used for accurate cost calculation
        
    Returns:
        AnalyticsTracker: Configured tracker, ready for use and attached to context
    """
    analytics = AnalyticsTracker()
    analytics.set_model(model_name)  # Configure for accurate cost calculation
    context.analytics = analytics    # Attach to context for sharing
    analytics.start_agent(agent_name)  # Begin tracking
    return analytics


@contextmanager
def analytics_context(context: "AnalysisContext", agent_name: str, model_name: str = DEFAULT_MODEL):
    """
    Context manager for analytics that ensures proper cleanup.
    
    This context manager provides automatic setup and teardown of analytics
    tracking, ensuring that analytics are always properly finished even if
    errors occur during analysis. It's particularly useful for ensuring
    robust error handling.
    
    Args:
        context (AnalysisContext): Context to attach analytics to
        agent_name (str): Name of the main agent
        model_name (str): AI model being used
        
    Yields:
        AnalyticsTracker: Configured and ready for use
        
    Example:
        with analytics_context(context, "Data Science Agent", model) as analytics:
            # Do analysis work...
            # Analytics will be automatically finished when exiting context
    """
    analytics = setup_analytics(context, agent_name, model_name)
    try:
        yield analytics
    finally:
        # Ensure analytics are always properly finished
        analytics.finish_agent(agent_name)
        analytics.finish()


def create_analytics_summary_event(analytics: AnalyticsTracker) -> str:
    """
    Create a formatted analytics summary for display in the UI.
    
    This function provides consistent formatting of analytics data across
    both agent systems for final results display. It creates a concise
    summary that highlights the key performance metrics users care about.
    
    Args:
        analytics (AnalyticsTracker): Tracker with completed analysis data
        
    Returns:
        str: Formatted string ready for UI display
        
    Example Output:
        "📊 Analytics: 45.2s, 8 tools, 3 phases, $0.0142"
    """
    summary = analytics.get_summary()
    return (
        f"📊 Analytics: {summary['total_duration']:.1f}s, "
        f"{summary['tool_calls']} tools, "
        f"{summary['phases_completed']} phases, "
        f"${summary['estimated_cost']:.4f}"
    )