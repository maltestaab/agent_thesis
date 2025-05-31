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
from data_science_agents.core.models import AgentResult, AnalysisResults, AnalysisMetrics
from data_science_agents.config.prompts import (
    BUSINESS_UNDERSTANDING_ENHANCED,
    DATA_UNDERSTANDING_ENHANCED,
    DATA_PREPARATION_ENHANCED,
    MODELING_ENHANCED,
    EVALUATION_ENHANCED,
    DEPLOYMENT_ENHANCED,
    ORCHESTRATOR_ENHANCED,
    CORE_INSTRUCTION
)
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS, MAX_TOKENS


@dataclass
class StreamingEvent:
    """Simple event for Streamlit updates"""
    event_type: str  # "text_delta", "tool_call", "tool_output", "agent_handoff", "message_complete", "analysis_complete"
    content: str
    timestamp: float
    agent_name: str = "Orchestrator"


class StreamEventBroadcaster:
    """Global event broadcaster for sub-agent streaming events"""
    def __init__(self):
        self.queue: Optional[asyncio.Queue] = None
    
    def set_queue(self, queue: asyncio.Queue):
        self.queue = queue
    
    async def broadcast_event(self, event: StreamingEvent):
        if self.queue:
            await self.queue.put(event)

# Global broadcaster instance
event_broadcaster = StreamEventBroadcaster()


def as_tool_enhanced(agent: Agent, tool_name: str, tool_description: str, max_turns: int = 25):
    """
    Enhanced version of as_tool() with max_turns and streaming support.
    
    Args:
        agent: The agent to convert to a tool
        tool_name: Name of the tool
        tool_description: Description of what the tool does
        max_turns: Maximum turns for the agent (default 25)
    """
    
    @function_tool(
        name_override=tool_name,
        description_override=tool_description,
    )
    async def run_agent_enhanced(context: RunContextWrapper, input: str) -> str:
        """Run agent with streaming and broadcasting support"""
        
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
        
        # Use run_streamed with higher max_turns
        result = Runner.run_streamed(
            agent,
            input=input,
            context=context.context,
            max_turns=max_turns
        )
        
        # Stream sub-agent events and broadcast them
        async for event in result.stream_events():
            # Check for cancellation - but don't yield, just return
            if getattr(st.session_state, 'cancel_analysis', False):
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
            
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                await event_broadcaster.broadcast_event(
                    StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time,
                        agent_name=agent.name
                    )
                )
                
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    await event_broadcaster.broadcast_event(
                        StreamingEvent(
                            event_type="tool_call",
                            content=f"🔧 {agent.name} executing code...",
                            timestamp=current_time,
                            agent_name=agent.name
                        )
                    )
                    
                elif event.item.type == "tool_call_output_item":
                    output_preview = str(event.item.output)[:150] + "..." if len(str(event.item.output)) > 150 else str(event.item.output)
                    await event_broadcaster.broadcast_event(
                        StreamingEvent(
                            event_type="tool_output",
                            content=f"📊 {agent.name} result:\n{output_preview}",
                            timestamp=current_time,
                            agent_name=agent.name
                        )
                    )
                    
                elif event.item.type == "message_output_item":
                    message_text = ItemHelpers.text_message_output(event.item)
                    if message_text.strip():
                        await event_broadcaster.broadcast_event(
                            StreamingEvent(
                                event_type="agent_reasoning",
                                content=f"💭 {agent.name}: {message_text}",
                                timestamp=current_time,
                                agent_name=agent.name
                            )
                        )
        
        # Broadcast completion
        if event_broadcaster.queue:
            await event_broadcaster.broadcast_event(
                StreamingEvent(
                    event_type="sub_agent_complete",
                    content=f"✅ {agent.name} completed",
                    timestamp=time.time(),
                    agent_name=agent.name
                )
            )
        
        return str(result.final_output)
    
    return run_agent_enhanced


# Model settings shared by all agents
model_settings = ModelSettings(
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P
)

# Create specialized CRISP-DM agents with context support and structured outputs
business_understanding_agent = Agent[AnalysisContext](
    name="Business Understanding Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=BUSINESS_UNDERSTANDING_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
)

data_understanding_agent = Agent[AnalysisContext](
    name="Data Understanding Agent", 
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=DATA_UNDERSTANDING_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
)

data_preparation_agent = Agent[AnalysisContext](
    name="Data Preparation Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=DATA_PREPARATION_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
)

modeling_agent = Agent[AnalysisContext](
    name="Modeling Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=MODELING_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
)

evaluation_agent = Agent[AnalysisContext](
    name="Evaluation Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=EVALUATION_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
)

deployment_agent = Agent[AnalysisContext](
    name="Deployment Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=DEPLOYMENT_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    output_type=AgentOutputSchema(output_type=AgentResult, strict_json_schema=False),
)

# Create orchestrator agent using enhanced as_tool functions
orchestration_agent = Agent[AnalysisContext](
    name="Data Science Orchestration Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=ORCHESTRATOR_ENHANCED.format(core_instruction=CORE_INSTRUCTION, max_turns=MAX_TURNS),
    tools=[
        # Use enhanced as_tool with higher max_turns and streaming
        as_tool_enhanced(
            business_understanding_agent,
            "business_understanding_agent",
            "Handles business understanding: defines objectives, success criteria, and project approach. Must specify input_file_used and output_file_created in response.",
            max_turns=MAX_TURNS
        ),
        as_tool_enhanced(
            data_understanding_agent,
            "data_understanding_agent",
            "Handles data understanding: loads data, explores structure, checks quality. Must specify input_file_used and output_file_created in response.",
            max_turns=MAX_TURNS
        ),
        as_tool_enhanced(
            data_preparation_agent,
            "data_preparation_agent",
            "Handles data preparation: cleans data, creates features, prepares for modeling. Must specify input_file_used and output_file_created in response.",
            max_turns=MAX_TURNS
        ),
        as_tool_enhanced(
            modeling_agent,
            "modeling_agent",
            "Handles modeling: builds and evaluates models. Must specify input_file_used and output_file_created in response.",
            max_turns=MAX_TURNS
        ),
        as_tool_enhanced(
            evaluation_agent,
            "evaluation_agent",
            "Handles evaluation: assesses results against business criteria, provides insights. Must specify input_file_used and output_file_created in response.",
            max_turns=MAX_TURNS
        ),
        as_tool_enhanced(
            deployment_agent,
            "deployment_agent",
            "Handles deployment planning: creates deployment strategy and documentation. Must specify input_file_used and output_file_created in response.",
            max_turns=MAX_TURNS
        )
    ]
)




async def run_multi_agent_analysis(prompt: str, file_name: str, max_turns: int = MAX_TURNS) -> AsyncGenerator[StreamingEvent, None]:
    """
    Run a complete data science analysis with full streaming visibility.
    
    Shows both orchestrator thinking and all sub-agent activities in real-time.
    """
    
    # Reset execution environment for clean start
    reset_execution_state()
    
    # Create analysis context
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="multi_agent",
        start_time=time.time()
    )
    
    # Create event queue for sub-agent events and connect broadcaster
    sub_agent_queue = asyncio.Queue()
    event_broadcaster.set_queue(sub_agent_queue)
    
    # Yield initial status
    yield StreamingEvent(
        event_type="analysis_start",
        content="🚀 Starting enhanced multi-agent analysis with full streaming...",
        timestamp=context.start_time
    )
    
    current_agent = "Orchestrator"
    
    try:
        with trace("Multi-Agent Data Science Analysis"):
            # Run orchestrator with streaming
            result = Runner.run_streamed(
                orchestration_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Simple direct streaming - no complex merging
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
            
            
            # Final completion event
            total_duration = time.time() - context.start_time
            images = get_created_images()
            
            yield StreamingEvent(
                event_type="analysis_complete",
                content=f"✅ Enhanced multi-agent analysis completed in {total_duration:.1f}s!\n\n📊 Final Results:\n{result.final_output}\n\n📸 Created {len(images)} visualizations",
                timestamp=time.time(),
                agent_name="Orchestrator"
            )
            
    except Exception as e:
        yield StreamingEvent(
            event_type="analysis_error",
            content=f"❌ Multi-agent analysis failed: {str(e)}",
            timestamp=time.time(),
            agent_name=current_agent
        )
    finally:
        # Clean up broadcaster
        event_broadcaster.set_queue(None)