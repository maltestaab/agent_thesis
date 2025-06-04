"""
data_science_agents/core/analytics.py - Analytics tracking with setup helpers

This module provides:
1. AnalyticsTracker class for comprehensive performance monitoring
2. Helper functions to eliminate setup duplication across agent systems
3. Model-aware cost calculation using pricing from settings.py
4. Single source of truth for all analysis metrics

Purpose: Centralizes all analytics functionality and provides helpers to eliminate
the duplicate analytics setup code that was repeated in both single_agent.py
and multi_agent.py systems.

Key Features:
- Tracks agent timing, tool usage, token consumption, and costs
- Model-specific cost calculation (different prices for different AI models)
- Integration with context system for state sharing
- Standardized setup/teardown patterns
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
    Tracks performance metrics for individual agents.
    
    In multi-agent mode, this lets us see which specialist agents
    take the most time and resources, helping optimize the system.
    
    Attributes:
        agent_name: Human-readable name (e.g., "Data Understanding Agent")
        start_time: Unix timestamp when agent started
        end_time: Unix timestamp when agent finished (None if still running)
        duration: Calculated runtime in seconds (None until finished)
        tool_calls: Number of tools this agent used (code execution, etc.)
    """
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tool_calls: int = 0
    
    def finish(self):
        """
        Mark this agent as finished and calculate final duration.
        Called when an agent completes its work.
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class AnalyticsTracker:
    """
    Comprehensive analytics tracking for data science analysis runs.
    
    This is the single source of truth for all performance metrics during
    an analysis. It tracks everything from basic timing to cost estimation
    and integrates with the broader system for real-time monitoring.
    
    Key Responsibilities:
    1. Track timing for overall analysis and individual agents
    2. Monitor tool usage (how many times agents executed code, etc.)
    3. Estimate token consumption and calculate costs using actual model pricing
    4. Provide real-time summaries for UI display
    5. Support different analysis modes (single vs multi-agent)
    
    Design Note: This does NOT track images - that's handled by execution.py
    to avoid duplication and maintain clear separation of concerns.
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
    # (OpenAI charges different rates for input vs output tokens)
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
        
        This is called when an agent starts work. In single-agent mode,
        there's just one call. In multi-agent mode, this is called for
        each specialist agent as they take control.
        
        Args:
            agent_name: Human-readable agent name for tracking
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
        
        Args:
            agent_name: Name of the agent that just finished
        """
        if agent_name in self.agent_timings:
            self.agent_timings[agent_name].finish()
            # Track completion for phase counting
            if agent_name not in self.phases_completed:
                self.phases_completed.append(agent_name)

    def estimate_tokens_from_content(self, content: str):
        """
        Estimate token usage from text content.
        
        This is called during streaming when we receive text from the AI model.
        We assume this is output content (agent responses) rather than input.
        
        Token estimation rule: Roughly 4 characters = 1 token (OpenAI's rule of thumb)
        
        Args:
            content: Text content from AI model response
        """
        estimated_tokens = len(content) / 4
        self.add_output_tokens(int(estimated_tokens))

    def add_input_tokens(self, token_count: int):
        """
        Add input tokens (user prompts, context, etc.) and update cost.
        
        Input tokens are what we send TO the AI model (prompts, data descriptions, etc.)
        These typically cost less than output tokens.
        
        Args:
            token_count: Number of input tokens to add
        """
        self.estimated_input_tokens += token_count
        self._update_cost_calculation()
        
    def add_output_tokens(self, token_count: int):
        """
        Add output tokens (AI responses) and update cost.
        
        Output tokens are what we get FROM the AI model (analysis, code, insights, etc.)
        These typically cost more than input tokens.
        
        Args:
            token_count: Number of output tokens to add
        """
        self.estimated_output_tokens += token_count
        self._update_cost_calculation()

    def _update_cost_calculation(self):
        """
        Update cost calculation using model-specific rates from settings.py.
        
        This is called automatically whenever tokens are added. It ensures
        that cost estimates are always current and use the correct pricing
        for whichever AI model the user selected.
        
        Why this matters: Different models have very different costs
        - gpt-4o-mini: $0.15 input, $0.60 output (per 1M tokens)
        - gpt-4o: $3.00 input, $10.00 output (per 1M tokens)
        """
        self.estimated_cost = get_model_cost(
            self.model_name, 
            self.estimated_input_tokens, 
            self.estimated_output_tokens
        )
        
    def add_tool_call(self, agent_name: str = None):
        """
        Record that an agent used a tool.
        
        Tools include things like executing Python code, calling other agents,
        searching the web, etc. This helps track how much work agents are doing.
        
        Args:
            agent_name: Which agent used the tool (for per-agent tracking)
        """
        self.total_tool_calls += 1
        if agent_name and agent_name in self.agent_timings:
            self.agent_timings[agent_name].tool_calls += 1
    
    def set_model(self, model_name: str):
        """
        Set the AI model being used and recalculate costs.
        
        This should be called when the analysis starts to ensure cost
        calculations use the correct pricing for the selected model.
        
        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
        """
        self.model_name = model_name
        self._update_cost_calculation()  # Recalculate with new model rates
    
    def finish(self):
        """
        Mark the entire analysis as completed and finalize all metrics.
        
        This should be called when the analysis is completely done,
        regardless of success or failure. It ensures all timing data
        is properly calculated and agents are marked as finished.
        """
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        # Finish any agent that's still running
        if self.current_agent and self.current_agent in self.agent_timings:
            self.agent_timings[self.current_agent].finish()
    
    def get_summary(self) -> Dict:
        """
        Get complete analytics summary for display and storage.
        
        This provides all the key metrics in a standardized format that
        can be used by the UI, stored in session state, or logged for
        system monitoring.
        
        Returns:
            Dictionary containing all analytics data:
            - total_duration: Overall analysis time
            - agent_durations: Per-agent timing breakdown
            - tool_calls: Total number of tool uses
            - phases_completed: Number of CRISP-DM phases finished
            - estimated_cost: Total cost in USD
            - token counts: Input/output token usage
            - images_created: Number of visualizations (from execution.py)
        """
        # Import here to avoid circular import issues
        from data_science_agents.core.execution import get_created_images
        
        # Calculate per-agent durations for breakdown analysis
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
# These functions eliminate the duplicate analytics setup code that was
# repeated in both single_agent.py and multi_agent.py. They provide a
# standardized way to initialize and manage analytics.

def setup_analytics(context: "AnalysisContext", agent_name: str, model_name: str = DEFAULT_MODEL) -> AnalyticsTracker:
    """
    Initialize analytics tracking and attach it to the analysis context.
    
    This helper eliminates the 4-5 lines of setup code that was duplicated
    in both agent systems. It creates the tracker, configures it properly,
    attaches it to the context for sharing, and starts tracking.
    
    Args:
        context: AnalysisContext object that will carry the analytics
        agent_name: Name of the main agent starting the analysis
        model_name: AI model being used (for accurate cost calculation)
        
    Returns:
        AnalyticsTracker ready for use, already attached to context
        
    Example:
        # Instead of duplicate setup in each agent system:
        analytics = AnalyticsTracker()
        context.analytics = analytics
        analytics.set_model(model)
        analytics.start_agent("Data Science Agent")
        
        # Use helper:
        analytics = setup_analytics(context, "Data Science Agent", model)
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
    
    This provides automatic setup and teardown of analytics tracking,
    ensuring that analytics are always properly finished even if
    errors occur during analysis.
    
    Args:
        context: AnalysisContext to attach analytics to
        agent_name: Name of the main agent
        model_name: AI model being used
        
    Yields:
        AnalyticsTracker configured and ready for use
        
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
    
    This provides consistent formatting of analytics data across
    both agent systems for the final results display.
    
    Args:
        analytics: AnalyticsTracker with completed analysis data
        
    Returns:
        Formatted string ready for UI display
        
    Example output:
        "📊 Analytics: 45.2s, 8 tools, 3 phases, $0.0142"
    """
    summary = analytics.get_summary()
    return (
        f"📊 Analytics: {summary['total_duration']:.1f}s, "
        f"{summary['tool_calls']} tools, "
        f"{summary['phases_completed']} phases, "
        f"${summary['estimated_cost']:.4f}"
    )