"""
data_science_agents/agent_systems/single_agent.py - Simplified single agent system

This module implements the single-agent approach to data science analysis where
one comprehensive agent handles all phases of the CRISP-DM methodology within
a single conversation flow.

SIMPLIFICATIONS MADE:
- Analytics helpers eliminate duplicate setup/teardown code
- Event helpers eliminate duplicate StreamingEvent creation  
- Centralized error handling with consistent patterns
- Standardized imports and configuration

The single agent approach offers:
1. Simpler orchestration (no inter-agent coordination needed)
2. Better context retention (one continuous conversation)
3. More flexible phase selection (can skip irrelevant phases)
4. Faster execution (no handoff overhead)

Trade-offs:
- Less specialized expertise per phase
- Potential for longer individual responses
- Single point of failure (if agent gets stuck)
"""
import time
import streamlit as st
from typing import AsyncGenerator
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers

# Core system imports
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext
from data_science_agents.core.analytics import setup_analytics, create_analytics_summary_event
from data_science_agents.core.events import (
    create_analysis_start_event, create_analysis_complete_event,
    create_error_event, create_cancellation_event, create_event,
    create_tool_call_event, create_tool_output_event
)

# Configuration imports
from data_science_agents.config.prompts import SINGLE_AGENT_ENHANCED
from data_science_agents.config.settings import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS_SINGLE
)


async def run_single_agent_analysis(
    prompt: str, 
    file_name: str, 
    max_turns: int = MAX_TURNS_SINGLE, 
    model: str = DEFAULT_MODEL
) -> AsyncGenerator:
    """
    Run comprehensive data science analysis using a single versatile agent.
    
    This function implements the single-agent approach where one AI agent
    handles the complete analysis workflow. The agent can intelligently
    decide which CRISP-DM phases to execute based on the specific request.
    
    Single Agent Benefits:
    1. Continuous context - Agent remembers everything from start to finish
    2. Flexible workflow - Can skip irrelevant phases or adapt the approach
    3. Simpler coordination - No need to manage handoffs between specialists
    4. Natural conversation flow - Analysis feels like a single coherent response
    
    Architecture:
    1. Single Data Science Agent - Handles all phases of analysis
    2. Execute Code Tool - Allows agent to run Python for data manipulation
    3. Context Management - Maintains state throughout the analysis
    4. Analytics Tracking - Monitors performance and costs
    5. Streaming Updates - Provides real-time progress to UI
    
    Args:
        prompt: User's analysis request describing what insights they want
        file_name: Name of the dataset file to analyze  
        max_turns: Maximum conversation turns (higher than multi-agent since one agent does everything)
        model: AI model to use (gpt-4o-mini, gpt-4o, etc.)
        
    Yields:
        StreamingEvent objects for real-time UI updates during analysis
        
    Process Flow:
    1. Initialize clean execution environment
    2. Create analysis context for state management
    3. Set up analytics tracking with model-aware costs
    4. Create single agent with comprehensive instructions
    5. Run agent with streaming updates and tool calls
    6. Handle completion, errors, and cleanup
    
    Turn Management:
    Single agents get a higher turn limit (500 vs 50 for specialists) because
    they need to complete the entire analysis in one conversation. Each turn
    includes the agent's reasoning plus any tool calls it makes.
    """
    
    # === INITIALIZATION ===
    # Reset execution environment to start with clean slate
    # This ensures no leftover variables from previous analyses
    reset_execution_state()
    
    # Create analysis context for state management and sharing
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="single_agent",
        start_time=time.time(),
        original_prompt=prompt
    )
    
    # Initialize analytics tracking with model-specific cost calculation
    analytics = setup_analytics(context, "Data Science Agent", model)
    
    # Announce analysis start to UI
    yield create_analysis_start_event("single_agent")

    # Analytics start notification (for UI metrics display)
    yield create_event(
        event_type="analytics_start",
        content="📊 Analytics tracking started",
        agent_name="System"
    )
    
    try:
        # Use OpenAI SDK's built-in tracing for monitoring and debugging
        with trace("Single Agent Data Science Analysis"):
            
            # === AGENT CREATION ===
            # Create the single comprehensive agent with full CRISP-DM capabilities
            data_science_agent = Agent(
                name="Data Science Agent",
                model=model,
                model_settings=ModelSettings(
                    temperature=DEFAULT_TEMPERATURE,  # Balanced creativity vs consistency
                    top_p=DEFAULT_TOP_P               # Token sampling strategy
                ),
                instructions=SINGLE_AGENT_ENHANCED,   # Comprehensive prompt with all phases
                tools=[execute_code]                  # Python execution capability
            )

            # === ANALYSIS EXECUTION ===
            # Run agent with streaming to provide real-time updates
            result = Runner.run_streamed(
                data_science_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # === STREAMING EVENT PROCESSING ===
            # Process events from the agent and convert to UI updates
            event_counter = 0
            async for event in result.stream_events():
                
                # Check for user cancellation (stop button clicked)
                if getattr(st.session_state, 'cancel_analysis', False):
                    result.cancel()
                    yield create_cancellation_event()
                    return
                
                current_time = time.time()
                event_counter += 1
                
                # === PERIODIC ANALYTICS UPDATES ===
                # Share analytics with streamlit every 10 events for live display
                if event_counter % 10 == 0 and analytics:
                    from data_science_agents.core.events import create_analytics_update_event
                    yield create_analytics_update_event(analytics)
                
                # Handle different types of events from the agent
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Token-by-token text streaming (like ChatGPT)
                    # Track tokens for cost estimation
                    analytics.estimate_tokens_from_content(event.data.delta)
                    
                    # Stream to UI for real-time typing effect
                    yield create_event(
                        event_type="text_delta",
                        content=event.data.delta,
                        agent_name="Data Science Agent"
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Agent is about to execute code or use a tool
                        analytics.add_tool_call("Data Science Agent")
                        
                        # Inform user that agent is taking action
                        yield create_tool_call_event(
                            "Agent is executing code to analyze data",
                            "Data Science Agent"
                        )
                        
                    elif event.item.type == "tool_call_output_item":
                        # Tool execution completed, show preview of results
                        output_text = str(event.item.output)
                        
                        # Show preview of output (first 300 chars for context)
                        preview = output_text[:300] + "..." if len(output_text) > 300 else output_text
                        
                        yield create_tool_output_event(
                            f"Code execution completed:\n{preview}",
                            "Data Science Agent"
                        )
                        
                    elif event.item.type == "message_output_item":
                        # Complete message from agent (reasoning/analysis)
                        message_text = ItemHelpers.text_message_output(event.item)
                        yield create_event(
                            event_type="agent_reasoning",
                            content=f"🤔 Agent reasoning: {message_text}",
                            agent_name="Data Science Agent"
                        )

            # === COMPLETION HANDLING ===
            # Finish analytics tracking
            analytics.finish_agent("Data Science Agent")
            analytics.finish()

            # Get created images for final summary
            images = get_created_images()
            # Note: Images are tracked by execution.py, not analytics (single source of truth)

            # Create analytics summary for display
            analytics_summary = create_analytics_summary_event(analytics)
            yield create_event(
                event_type="analytics_complete",
                content=analytics_summary,
                agent_name="System"
            )

            # Calculate total duration and create completion event
            total_duration = time.time() - context.start_time
            
            yield create_analysis_complete_event(
                final_output=result.final_output,
                duration=total_duration,
                image_count=len(images)
            )
            
    except Exception as e:
        # === ERROR HANDLING ===
        # Ensure analytics cleanup even on error
        analytics.finish_agent("Data Science Agent")
        analytics.finish()
        
        # Inform user of the error with standardized format
        yield create_error_event(f"Analysis failed: {str(e)}")
        
    finally:
        # === CLEANUP ===
        # Guarantee analytics is always properly finished
        # This ensures cost tracking is accurate even if errors occur
        if analytics and not analytics.end_time:
            analytics.finish()