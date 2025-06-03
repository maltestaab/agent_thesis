"""
data_science_agents/agent_systems/multi_agent.py - Enhanced multi-agent with full streaming
"""
import time
import streamlit as st
import asyncio
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers, AgentOutputSchema, function_tool, RunContextWrapper
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext 
from data_science_agents.core.models import AgentResult
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS
from data_science_agents.core.events import StreamingEvent
from data_science_agents.core.analytics import AnalyticsTracker
from data_science_agents.config.prompts import (
    BUSINESS_UNDERSTANDING_ENHANCED,
    DATA_UNDERSTANDING_ENHANCED,
    DATA_PREPARATION_ENHANCED,
    MODELING_ENHANCED,
    EVALUATION_ENHANCED,
    DEPLOYMENT_ENHANCED,
    ORCHESTRATOR_ENHANCED,
)




class StreamEventBroadcaster:
    """Global event broadcaster for sub-agent streaming events""" # All agents can send their updates here, and the main system receives them all.
    def __init__(self): # Queue for updates
        self.queue: Optional[asyncio.Queue] = None
    
    def set_queue(self, queue: asyncio.Queue): # Connecting broadcaster to queue
        self.queue = queue
    
    async def broadcast_event(self, event: StreamingEvent): # Adding event to the queue
        if self.queue:
            await self.queue.put(event)

# Global broadcaster instantiation
event_broadcaster = StreamEventBroadcaster()


def as_tool_enhanced(agent: Agent, tool_name: str, tool_description: str, max_turns: int = 25): # "Orchestrator can "call" the subagents to do tasks."
    """Adaptation of the as_tool source code to handle streaming events and max_turns for the subagents"""
    
    def is_json_like(text: str) -> bool:
        """Check if text looks like JSON structured output (to exclude them from the streaming in streamlit)"""
        text = text.strip()
        return (text.startswith('{') and 
                'phase' in text and 
                'summary' in text and
                'key_findings' in text)
    
    @function_tool(
        name_override=tool_name,
        description_override=tool_description,
    )
    async def run_agent_enhanced(context: RunContextWrapper, input: str) -> str: # Updated version of running the subagents inside the as_tool function to handle streaming
        """Run agent with streaming and reasoning filtering"""
        
        # Track agent start in analytics if available
        if hasattr(context.context, 'analytics'):
            context.context.analytics.start_agent(agent.name)
        
        # Broadcast that this sub-agent is starting
        if event_broadcaster.queue:
            await event_broadcaster.broadcast_event(
                StreamingEvent(
                    event_type="sub_agent_start",
                    content=f"🤖 Starting {agent.name}...",
                    timestamp=time.time(),
                    agent_name=agent.name
                )
            )
        
        result = Runner.run_streamed(
            agent,
            input=input,
            context=context.context,
            max_turns=max_turns
        )
        
        # Stream sub-agent events with JSON filtering
        async for event in result.stream_events(): # Process all events from the subagent
            # Check for cancellation
            if getattr(st.session_state, 'cancel_analysis', False): # Checks if user has clicked cancel button
                result.cancel()
                if event_broadcaster.queue:
                    await event_broadcaster.broadcast_event(
                        StreamingEvent(
                            event_type="analysis_cancelled",
                            content="🛑 Analysis cancelled by user",
                            timestamp=time.time(),
                            agent_name=agent.name
                        )
                    )
                return "Analysis cancelled by user"
            
            if not event_broadcaster.queue:
                continue
                
            current_time = time.time()
            
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent): # word-by-word streaming of Agent thinking
                if hasattr(context.context, 'analytics'):
                    context.context.analytics.estimate_tokens_from_content(event.data.delta)
    
                await event_broadcaster.broadcast_event(
                    StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time,
                        agent_name=agent.name
                    )
                )
                
            elif event.type == "run_item_stream_event": # Streaming of the subagent's tool calls and outputs
                if event.item.type == "tool_call_item":
                    # Track tool call in analytics
                    if hasattr(context.context, 'analytics'):
                        context.context.analytics.add_tool_call(agent.name)
                    
                    await event_broadcaster.broadcast_event(
                        StreamingEvent(
                            event_type="tool_call",
                            content=f"🔧 {agent.name} executing code...",
                            timestamp=current_time,
                            agent_name=agent.name
                        )
                    )
                    
                elif event.item.type == "tool_call_output_item": # Code execution output
                    output_preview = str(event.item.output)[:150] + "..." if len(str(event.item.output)) > 150 else str(event.item.output)
                    await event_broadcaster.broadcast_event(
                        StreamingEvent(
                            event_type="tool_output",
                            content=f"📊 {agent.name} result:\n{output_preview}",
                            timestamp=current_time,
                            agent_name=agent.name
                        )
                    )
                    
                elif event.item.type == "message_output_item": # Agent complete thought/message
                    # FILTER JSON FROM REASONING
                    message_text = ItemHelpers.text_message_output(event.item)
                    
                    # Skip if this looks like structured JSON output (to keep streaming clean in streamlit)
                    if is_json_like(message_text):
                        # Instead of showing raw JSON, show a clean completion message
                        await event_broadcaster.broadcast_event(
                            StreamingEvent(
                                event_type="agent_result",
                                content=f"✅ {agent.name} completed analysis and provided structured results",
                                timestamp=current_time,
                                agent_name=agent.name
                            )
                        )
                    elif message_text.strip():
                        # Actual reasoning part of the agent's output that is not JSON
                        await event_broadcaster.broadcast_event(
                            StreamingEvent(
                                event_type="agent_reasoning",
                                content=f"🤔 {agent.name}: {message_text}",
                                timestamp=current_time,
                                agent_name=agent.name
                            )
                        )
        
        # Finish agent tracking in analytics
        if hasattr(context.context, 'analytics'):
            context.context.analytics.finish_agent(agent.name)
        
        # Broadcast completion. Tells that specialist agent has completed their task.
        if event_broadcaster.queue:
            await event_broadcaster.broadcast_event(
                StreamingEvent(
                    event_type="sub_agent_complete",
                    content=f"✅ {agent.name} completed",
                    timestamp=time.time(),
                    agent_name=agent.name
                )
            )
        
        return str(result.final_output) # Return the final output of the subagent to the orchestrator
    
    return run_agent_enhanced


async def run_multi_agent_analysis(prompt: str, file_name: str, max_turns: int = MAX_TURNS, model: str = DEFAULT_MODEL) -> AsyncGenerator[StreamingEvent, None]:
    """
    Run a complete data science analysis with full streaming visibility. (Run on click in streamlit)
    
    Shows both orchestrator thinking and all sub-agent activities in real-time.
    """
    
    # Reset execution environment for clean start
    reset_execution_state()
    
    # Create analysis context
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="multi_agent",
        start_time=time.time(),
        original_prompt=prompt
    )
    
    # Initialize analytics tracking
    analytics = AnalyticsTracker()
    context.analytics = analytics  # Attach to context
    analytics.start_agent("Orchestrator")
    
    # Create event queue for sub-agent events and connect broadcaster
    sub_agent_queue = asyncio.Queue()
    event_broadcaster.set_queue(sub_agent_queue)
    
    # Yield initial status
    yield StreamingEvent(
        event_type="analysis_start",
        content="🚀 Starting enhanced multi-agent analysis with full streaming...",
        timestamp=context.start_time
    )
    
    # Yield analytics start event
    yield StreamingEvent(
        event_type="analytics_start",
        content=f"📊 Analytics tracking started",
        timestamp=time.time(),
        agent_name="System"
    )
    
    current_agent = "Orchestrator"
    
    try:
        with trace("Multi-Agent Data Science Analysis"):
            # Create agents with selected model
            model_settings = ModelSettings(
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )

            # Create specialized CRISP-DM agents with selected model
            business_understanding_agent = Agent[AnalysisContext](
                name="Business Understanding Agent",
                model=model,
                model_settings=model_settings,
                instructions=BUSINESS_UNDERSTANDING_ENHANCED,
                tools=[execute_code],
                output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
            )

            data_understanding_agent = Agent[AnalysisContext](
                name="Data Understanding Agent", 
                model=model,
                model_settings=model_settings,
                instructions=DATA_UNDERSTANDING_ENHANCED,
                tools=[execute_code],
                output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
            )

            data_preparation_agent = Agent[AnalysisContext](
                name="Data Preparation Agent",
                model=model,
                model_settings=model_settings,
                instructions=DATA_PREPARATION_ENHANCED,
                tools=[execute_code],
                output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
            )

            modeling_agent = Agent[AnalysisContext](
                name="Modeling Agent",
                model=model,
                model_settings=model_settings,
                instructions=MODELING_ENHANCED,
                tools=[execute_code],
                output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
            )

            evaluation_agent = Agent[AnalysisContext](
                name="Evaluation Agent",
                model=model,
                model_settings=model_settings,
                instructions=EVALUATION_ENHANCED,
                tools=[execute_code],
                output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
            )

            deployment_agent = Agent[AnalysisContext](
                name="Deployment Agent",
                model=model,
                model_settings=model_settings,
                instructions=DEPLOYMENT_ENHANCED,
                tools=[execute_code],
                output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
            )

            # Create orchestrator agent with selected model
            orchestration_agent = Agent[AnalysisContext](
                name="Data Science Orchestration Agent",
                model=model,
                model_settings=model_settings,
                instructions=ORCHESTRATOR_ENHANCED.format(max_turns=MAX_TURNS),
                tools=[
                    as_tool_enhanced(business_understanding_agent, "business_understanding_agent", "Handles business understanding: defines objectives, success criteria, and project approach. Must specify input_file_used and output_file_created in response.", max_turns=MAX_TURNS),
                    as_tool_enhanced(data_understanding_agent, "data_understanding_agent", "Handles data understanding: loads data, explores structure, checks quality. Must specify input_file_used and output_file_created in response.", max_turns=MAX_TURNS),
                    as_tool_enhanced(data_preparation_agent, "data_preparation_agent", "Handles data preparation: cleans data, creates features, prepares for modeling. Must specify input_file_used and output_file_created in response.", max_turns=MAX_TURNS),
                    as_tool_enhanced(modeling_agent, "modeling_agent", "Handles modeling: builds and evaluates models. Must specify input_file_used and output_file_created in response.", max_turns=MAX_TURNS),
                    as_tool_enhanced(evaluation_agent, "evaluation_agent", "Handles evaluation: assesses results against business criteria, provides insights. Must specify input_file_used and output_file_created in response.", max_turns=MAX_TURNS),
                    as_tool_enhanced(deployment_agent, "deployment_agent", "Handles deployment planning: creates deployment strategy and documentation. Must specify input_file_used and output_file_created in response.", max_turns=MAX_TURNS)
                ]
            )

            # Run orchestrator with streaming
            result = Runner.run_streamed(
                orchestration_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Simple direct streaming of the events
            async for event in result.stream_events():
                current_time = time.time()
                
                # Handle orchestrator events
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    yield StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time,
                        agent_name=current_agent
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Track orchestrator tool calls (calling sub-agents)
                        analytics.add_tool_call("Orchestrator")
                        
                        tool_name = getattr(event.item, 'name', 'unknown_tool')
                        if 'agent' in tool_name:
                            agent_display_name = tool_name.replace('_agent', ' Agent').title()
                            yield StreamingEvent(
                                event_type="tool_call",
                                content=f"🤖 Orchestrator calling {agent_display_name}...",
                                timestamp=current_time,
                                agent_name=current_agent
                            )
                    
                    elif event.item.type == "tool_call_output_item":
                        yield StreamingEvent(
                            event_type="tool_output",
                            content="🔄 Sub-agent completed, orchestrator continuing...",
                            timestamp=current_time,
                            agent_name=current_agent
                        )
                
                # Check for sub-agent events (non-blocking)
                try:
                    while True:
                        sub_event = sub_agent_queue.get_nowait()
                        yield sub_event
                except asyncio.QueueEmpty:
                    pass
            
            # Finish analytics tracking
            analytics.finish_agent("Orchestrator")
            analytics.finish()

            # Get images for analytics
            images = get_created_images()
            for img in images:
                analytics.add_image(img)

            # Yield analytics summary
            analytics_summary = analytics.get_summary()
            yield StreamingEvent(
                event_type="analytics_complete",
                content=f"📊 Analytics: {analytics_summary['total_duration']:.1f}s, {analytics_summary['tool_calls']} tools, {analytics_summary['phases_completed']} phases, ${analytics_summary['estimated_cost']:.4f}",
                timestamp=time.time(),
                agent_name="System"
            )
            
            # Final completion event
            total_duration = time.time() - context.start_time
            
            yield StreamingEvent(
                event_type="analysis_complete",
                content=f"✅ Enhanced multi-agent analysis completed in {total_duration:.1f}s!\n\n📊 Final Results:\n{result.final_output}\n\n📸 Created {len(images)} visualizations",
                timestamp=time.time(),
                agent_name="Orchestrator"
            )
            
    except Exception as e:
        # Finish analytics even on error
        analytics.finish_agent("Orchestrator")
        analytics.finish()
        
        yield StreamingEvent(
            event_type="analysis_error",
            content=f"❌ Multi-agent analysis failed: {str(e)}",
            timestamp=time.time(),
            agent_name=current_agent
        )
    finally:
        # Ensure analytics is always finished
        if analytics and not analytics.end_time:
            analytics.finish()
        
        # Clean up broadcaster
        event_broadcaster.set_queue(None)