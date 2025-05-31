"""
data_science_agents/agent_systems/single_agent.py - Single agent system with flexible CRISP-DM
"""
import time
import streamlit as st

from typing import AsyncGenerator
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext
from data_science_agents.core.models import AnalysisResults, AnalysisMetrics
from data_science_agents.config.prompts import SINGLE_AGENT_ENHANCED, CORE_INSTRUCTION
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS, MAX_TOKENS


@dataclass
class StreamingEvent:
    """Simple event for Streamlit updates"""
    event_type: str  # "text_delta", "tool_call", "tool_output", "message_complete", "analysis_complete"
    content: str
    timestamp: float
    agent_name: str = "Data Science Agent"


# Create the single data science agent with context support
# This agent handles all CRISP-DM phases in one comprehensive workflow
data_science_agent = Agent[AnalysisContext](
    name="Data Science Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P
    ),
    instructions=SINGLE_AGENT_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code]
)


async def run_single_agent_analysis(prompt: str, file_name: str, max_turns: int = MAX_TURNS) -> AsyncGenerator[StreamingEvent, None]:
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
        
    Yields:
        StreamingEvent objects for real-time UI updates
    """
    
    # Reset execution environment for clean start
    reset_execution_state()
    
    # Create analysis context to prevent redundant work
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="single_agent",
        start_time=time.time()
    )
    
    # Yield initial status
    yield StreamingEvent(
        event_type="analysis_start",
        content="🚀 Starting single agent analysis...",
        timestamp=context.start_time
    )
    
    try:
        # Use SDK's built-in tracing for proper monitoring
        with trace("Single Agent Data Science Analysis"):
            # Run agent with streaming using the SDK's native runner
            result = Runner.run_streamed(
                data_science_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Stream events to Streamlit in real-time
            async for event in result.stream_events():
                if getattr(st.session_state, 'cancel_analysis', False):
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
                    # Token-by-token text streaming - like ChatGPT
                    yield StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
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

            # Final completion event with clean results
            total_duration = time.time() - context.start_time
            images = get_created_images()

            # Don't include the streaming content in final output - show clean final result
            yield StreamingEvent(
                event_type="analysi