"""
data_science_agents/agent_systems/single_agent.py - Single Agent Analysis System

This module implements a single-agent approach to data science analysis where
one comprehensive AI agent handles all phases of the CRISP-DM methodology:

- Business Understanding: Define objectives and success criteria
- Data Understanding: Explore and analyze the dataset structure
- Data Preparation: Clean and prepare data for analysis
- Modeling: Build and evaluate machine learning models
- Evaluation: Assess results and provide business insights
- Deployment: Provide implementation guidance

The single agent approach offers:
- Simpler orchestration with no inter-agent coordination needed
- Better context retention throughout the analysis
- Flexible phase selection based on the specific request
- Faster execution with no handoff overhead

Key Features:
- Comprehensive CRISP-DM methodology implementation
- Real-time streaming of analysis progress
- Flexible phase execution based on task requirements
- Advanced analytics tracking for performance monitoring
"""

import time
import streamlit as st
from typing import AsyncGenerator
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers

# Import core system components
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext
from data_science_agents.core.analytics import setup_analytics, create_analytics_summary_event
from data_science_agents.core.events import (
    create_analysis_start_event, create_analysis_complete_event,
    create_error_event, create_cancellation_event, create_event,
    create_tool_call_event, create_tool_output_event
)

# Import configuration settings
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
    Execute comprehensive data science analysis using a single AI agent.
    
    This function orchestrates a complete data science workflow using one
    versatile agent that can handle all phases of the CRISP-DM methodology.
    The agent intelligently determines which phases are needed based on the
    user's specific request and executes them in a logical sequence.
    
    Architecture Components:
    1. Single Data Science Agent: Handles all analysis phases autonomously
    2. Execute Code Tool: Enables Python execution for data manipulation
    3. Context Management: Maintains analysis state throughout execution
    4. Analytics Tracking: Monitors performance metrics and costs
    5. Streaming Updates: Provides real-time progress to the user interface
    
    Args:
        prompt (str): User's analysis request describing desired insights
        file_name (str): Name of the dataset file to analyze
        max_turns (int): Maximum conversation turns (default from settings)
        model (str): AI model to use (e.g., 'gpt-4o-mini', 'gpt-4o')
        
    Yields:
        StreamingEvent: Real-time events for UI updates during analysis
        
    Process Flow:
    1. Initialize clean execution environment
    2. Create analysis context for state management
    3. Set up analytics tracking with model-specific costs
    4. Create and configure the data science agent
    5. Execute analysis with real-time streaming
    6. Handle completion, errors, and cleanup
    """
    
    # === ENVIRONMENT INITIALIZATION ===
    # Reset the execution environment to ensure clean start
    # This clears any variables from previous analyses
    reset_execution_state()
    
    # Create analysis context to track state throughout the process
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="single_agent",
        start_time=time.time(),
        original_prompt=prompt
    )
    
    # Initialize analytics tracking with model-specific cost calculation
    analytics = setup_analytics(context, "Data Science Agent", model)
    
    # Notify the user interface that analysis is starting
    yield create_analysis_start_event("single_agent")

    # Send analytics initialization event for UI metrics display
    yield create_event(
        event_type="analytics_start",
        content="📊 Analytics tracking started",
        agent_name="System"
    )
    
    try:
        # Use OpenAI SDK's built-in tracing for monitoring and debugging
        with trace("Single Agent Data Science Analysis"):
            
            # === AGENT CREATION ===
            # Create the comprehensive data science agent
            data_science_agent = Agent(
                name="Data Science Agent",
                model=model,
                model_settings=ModelSettings(
                    temperature=DEFAULT_TEMPERATURE,  # Balanced creativity vs consistency
                    top_p=DEFAULT_TOP_P               # Token sampling strategy
                ),
                instructions=SINGLE_AGENT_ENHANCED,   # Comprehensive analysis prompts
                tools=[execute_code]                  # Python execution capability
            )

            # === ANALYSIS EXECUTION ===
            # Run the agent with streaming to provide real-time updates
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
                    # Token-by-token text streaming (similar to ChatGPT interface)
                    # Track tokens for accurate cost estimation
                    analytics.estimate_tokens_from_content(event.data.delta)
                    
                    # Stream text to UI for real-time typing effect
                    yield create_event(
                        event_type="text_delta",
                        content=event.data.delta,
                        agent_name="Data Science Agent"
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Agent is about to execute code or use a tool
                        analytics.add_tool_call("Data Science Agent")
                        
                        # Notify user that agent is taking action
                        yield create_tool_call_event(
                            "Agent is executing code to analyze data",
                            "Data Science Agent"
                        )
                        
                    elif event.item.type == "tool_call_output_item":
                        # Tool execution completed, show preview of results
                        output_text = str(event.item.output)
                        
                        # Create preview of output (first 300 characters for context)
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
            # Finalize analytics tracking
            analytics.finish_agent("Data Science Agent")
            analytics.finish()

            # Retrieve created images for final summary
            images = get_created_images()

            # Create analytics summary for display
            analytics_summary = create_analytics_summary_event(analytics)
            yield create_event(
                event_type="analytics_complete",
                content=analytics_summary,
                agent_name="System"
            )

            # Calculate total analysis duration
            total_duration = time.time() - context.start_time
            
            # Send completion event with final results
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
        
        # Inform user of the error with standardized formatting
        yield create_error_event(f"Analysis failed: {str(e)}")
        
    finally:
        # === CLEANUP ===
        # Guarantee analytics is always properly finished
        # This ensures cost tracking is accurate even if errors occur
        if analytics and not analytics.end_time:
            analytics.finish()