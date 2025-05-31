"""
data_science_agents/agent_systems/multi_agent.py - Multi-agent system with streaming
"""
import time
from typing import AsyncGenerator
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers, AgentOutputSchema
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


# Model settings shared by all agents
model_settings = ModelSettings(
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P
)

# Create specialized CRISP-DM agents with context support and structured outputs
# Each agent is responsible for one phase and returns a structured AgentResult

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

# Create orchestrator agent using the SDK's native agent-as-tool feature
# This replaces the complex custom wrapper functions with simple .as_tool() calls
orchestration_agent = Agent[AnalysisContext](
    name="Data Science Orchestration Agent",
    model=DEFAULT_MODEL,
    model_settings=model_settings,
    instructions=ORCHESTRATOR_ENHANCED.format(core_instruction=CORE_INSTRUCTION),
    tools=[
        # Use the SDK's native .as_tool() method for each specialist agent
        # This is much simpler than the previous custom wrapper approach
        business_understanding_agent.as_tool(
            tool_name="business_understanding_agent",
            tool_description="Handles business understanding: defines objectives, success criteria, and project approach. Returns structured AgentResult."
        ),
        data_understanding_agent.as_tool(
            tool_name="data_understanding_agent",
            tool_description="Handles data understanding: loads data, explores structure, checks quality. Returns structured AgentResult with data info."
        ),
        data_preparation_agent.as_tool(
            tool_name="data_preparation_agent",
            tool_description="Handles data preparation: cleans data, creates features, prepares for modeling. Returns structured AgentResult."
        ),
        modeling_agent.as_tool(
            tool_name="modeling_agent",
            tool_description="Handles modeling: builds and evaluates models. Returns structured AgentResult with performance metrics."
        ),
        evaluation_agent.as_tool(
            tool_name="evaluation_agent",
            tool_description="Handles evaluation: assesses results against business criteria, provides insights. Returns structured AgentResult."
        ),
        deployment_agent.as_tool(
            tool_name="deployment_agent",
            tool_description="Handles deployment planning: creates deployment strategy and documentation. Returns structured AgentResult."
        )
    ]
)


async def run_multi_agent_analysis(prompt: str, file_name: str, max_turns: int = MAX_TURNS) -> AsyncGenerator[StreamingEvent, None]:
    """
    Run a complete data science analysis using multiple specialized agents with streaming.
    
    This function:
    1. Resets the execution environment to start clean
    2. Creates analysis context to share state between agents
    3. Runs the orchestrator agent which manages all specialist agents
    4. Yields real-time events showing agent handoffs and work
    
    Args:
        prompt: The analysis request/prompt
        file_name: Name of the data file being analyzed
        max_turns: Maximum number of turns for the orchestrator
        
    Yields:
        StreamingEvent objects for real-time UI updates
    """
    
    # Reset execution environment for clean start
    reset_execution_state()
    
    # Create analysis context to coordinate between agents
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="multi_agent",
        start_time=time.time()
    )
    
    # Yield initial status
    yield StreamingEvent(
        event_type="analysis_start",
        content="🚀 Starting multi-agent analysis...",
        timestamp=context.start_time
    )
    
    current_agent = "Orchestrator"
    
    try:
        # Use SDK's built-in tracing for proper monitoring
        with trace("Multi-Agent Data Science Analysis"):
            # Run orchestrator with streaming - it will call specialist agents as needed
            result = Runner.run_streamed(
                orchestration_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Stream events to Streamlit in real-time
            async for event in result.stream_events():
                current_time = time.time()
                
                # Handle different event types for live updates
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Token-by-token text streaming from current agent
                    yield StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time,
                        agent_name=current_agent
                    )
                    
                elif event.type == "agent_updated_stream_event":
                    # Agent handoff occurred - show user which agent is now active
                    current_agent = event.new_agent.name
                    yield StreamingEvent(
                        event_type="agent_handoff",
                        content=f"🔄 Switched to: {current_agent}",
                        timestamp=current_time,
                        agent_name=current_agent
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Tool/agent being called - show what's happening
                        tool_name = getattr(event.item, 'name', 'execute_code')
                        if 'agent' in tool_name:
                            # Specialist agent being called
                            agent_display_name = tool_name.replace('_agent', ' Agent').title()
                            yield StreamingEvent(
                                event_type="tool_call",
                                content=f"🤖 Calling {agent_display_name}...",
                                timestamp=current_time,
                                agent_name=current_agent
                            )
                        else:
                            # Code execution tool being called
                            yield StreamingEvent(
                                event_type="tool_call",
                                content="🔧 Executing code...",
                                timestamp=current_time,
                                agent_name=current_agent
                            )
                        
                    elif event.item.type == "tool_call_output_item":
                        # Tool/agent execution completed - show results preview
                        output_preview = str(event.item.output)[:200] + "..." if len(str(event.item.output)) > 200 else str(event.item.output)
                        yield StreamingEvent(
                            event_type="tool_output",
                            content=f"📊 Result:\n{output_preview}",
                            timestamp=current_time,
                            agent_name=current_agent
                        )
                        
                    elif event.item.type == "message_output_item":
                        # Complete message from agent
                        message_text = ItemHelpers.text_message_output(event.item)
                        yield StreamingEvent(
                            event_type="message_complete",
                            content=f"💭 {current_agent}: {message_text}",
                            timestamp=current_time,
                            agent_name=current_agent
                        )
            
            # Final completion event with results
            total_duration = time.time() - context.start_time
            images = get_created_images()
            
            yield StreamingEvent(
                event_type="analysis_complete",
                content=f"✅ Multi-agent analysis completed in {total_duration:.1f}s!\n\n📊 Final Results:\n{result.final_output}\n\n📸 Created {len(images)} visualizations",
                timestamp=time.time(),
                agent_name="Orchestrator"
            )
            
    except Exception as e:
        # Simple error handling - yield error event
        yield StreamingEvent(
            event_type="analysis_error",
            content=f"❌ Multi-agent analysis failed: {str(e)}",
            timestamp=time.time(),
            agent_name=current_agent
        )