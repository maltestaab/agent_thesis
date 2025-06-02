"""
data_science_agents/agent_systems/single_agent.py - Single agent system
"""
import time
import streamlit as st

from typing import AsyncGenerator
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext
from data_science_agents.config.prompts import SINGLE_AGENT_ENHANCED, CORE_INSTRUCTION
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS_SINGLE, MAX_TOKENS
from data_science_agents.core.events import StreamingEvent
from data_science_agents.core.analytics import AnalyticsTracker


async def run_single_agent_analysis(prompt: str, file_name: str, max_turns: int = MAX_TURNS_SINGLE, model: str = DEFAULT_MODEL) -> AsyncGenerator[StreamingEvent, None]:
    """
    Run a complete data science analysis using a single agent with streaming.
    
    This function:
    1. Resets the execution environment to start clean
    2. Creates analysis context to avoid redundant work
    3. Runs the single agent with streaming updates
    4. Yields real-time events for Streamlit display
    
    Args:
        prompt: The analysis request/prompt
        file_name: Name of the data file being analyzed
        max_turns: Maximum number of turns for the agent
        model: AI model to use for analysis
        
    Yields:
        StreamingEvent objects for real-time UI updates
    """
    
    # Reset execution environment for clean start
    reset_execution_state()
    
    # Create analysis context to prevent redundant work (state management)
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="single_agent",
        start_time=time.time()
    )
    
    # Initialize analytics tracking
    analytics = AnalyticsTracker()
    analytics.start_agent("Data Science Agent")
    
    # Yield initial status
    yield StreamingEvent(
        event_type="analysis_start",
        content="🚀 Starting single agent analysis...",
        timestamp=context.start_time
    )

    # Yield analytics start event
    yield StreamingEvent(
        event_type="analytics_start",
        content=f"📊 Analytics tracking started",
        timestamp=time.time(),
        agent_name="System"
    )
    
    try:
        # Use SDK's built-in tracing for proper monitoring
        with trace("Single Agent Data Science Analysis"):
            # Create agent with selected model
            data_science_agent = Agent[AnalysisContext](
                name="Data Science Agent",
                model=model,
                model_settings=ModelSettings(
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
                ),
                instructions=SINGLE_AGENT_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
                tools=[execute_code]
            )

            # Run agent with streaming using the SDK's native runner
            result = Runner.run_streamed(
                data_science_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Stream events to Streamlit in real-time
            async for event in result.stream_events():
                if getattr(st.session_state, 'cancel_analysis', False): # checks if user has clicked cancel button
                    result.cancel()
                    yield StreamingEvent(
                        event_type="analysis_cancelled",
                        content="🛑 Analysis cancelled by user",
                        timestamp=time.time()
                    )
                    return
                current_time = time.time()
                
                # Handle different event types for live updates
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    analytics.estimate_tokens_from_content(event.data.delta)
                    # Token-by-token text streaming - like ChatGPT
                    yield StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Track tool call in analytics
                        analytics.add_tool_call("Data Science Agent")
                        
                        # Tool being called - show user what's happening with more detail
                        yield StreamingEvent(
                            event_type="tool_call",
                            content="🔧 Agent is executing code to analyze data...",
                            timestamp=current_time
                        )
                        
                    elif event.item.type == "tool_call_output_item":
                        # Tool execution completed - show results with more context
                        output_text = str(event.item.output)
                        # Show first 300 chars for context
                        preview = output_text[:300] + "..." if len(output_text) > 300 else output_text
                        yield StreamingEvent(
                            event_type="tool_output",
                            content=f"📊 Code execution completed:\n{preview}",
                            timestamp=current_time
                        )
                        
                    elif event.item.type == "message_output_item":
                        # Complete message from agent - this is the thinking/reasoning
                        message_text = ItemHelpers.text_message_output(event.item)
                        yield StreamingEvent(
                            event_type="agent_reasoning",
                            content=f"🤔 Agent reasoning: {message_text}",
                            timestamp=current_time
                        )

            # Finish analytics tracking
            analytics.finish_agent("Data Science Agent")
            analytics.finish()

            # Get images for analytics
            images = get_created_images()
            for img in images:
                analytics.add_image(img)

            # Yield analytics summary
            analytics_summary = analytics.get_summary()
            yield StreamingEvent(
                event_type="analytics_complete",
                content=f"📊 Analytics: {analytics_summary['total_duration']:.1f}s, {analytics_summary['tool_calls']} tools, ${analytics_summary['estimated_cost']:.4f}",
                timestamp=time.time(),
                agent_name="System"
            )

            # Final completion event with clean results
            total_duration = time.time() - context.start_time

            # Don't include the streaming content in final output - show clean final result
            yield StreamingEvent(
                event_type="analysis_complete",
                content=result.final_output,  # Just the clean final output
                timestamp=time.time()
            )
            
    except Exception as e:
        # Finish analytics even on error
        analytics.finish_agent("Data Science Agent")
        analytics.finish()
        
        # Simple error handling - yield error event
        yield StreamingEvent(
            event_type="analysis_error",
            content=f"❌ Analysis failed: {str(e)}",
            timestamp=time.time()
        )
    finally:
        # Ensure analytics is always finished
        if analytics and not analytics.end_time:
            analytics.finish()