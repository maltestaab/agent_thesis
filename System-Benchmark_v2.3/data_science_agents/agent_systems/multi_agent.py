"""
data_science_agents/agent_systems/multi_agent.py - Simple multi-agent with clean output display
"""
import time
import streamlit as st
import asyncio
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, ItemHelpers, function_tool, RunContextWrapper
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext 
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS, MAX_TURNS_SPECIALIST
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


def create_full_context_for_specialist(user_request: str, completed_phases: dict, current_task: str, data_file: str = None, context: AnalysisContext = None) -> str:
    """Create comprehensive context for specialist agents"""
    
    context_parts = [
        f"ANALYSIS REQUEST: {user_request}",
        f"ORIGINAL DATASET: {data_file}" if data_file else ""
    ]
    
    # Add available variables/dataframes information
    if context:
        available_vars = context.get_available_variables()
        if available_vars:
            dataframe_vars = [name for name, desc in available_vars.items() 
                            if 'DataFrame' in desc or 'pandas' in desc.lower()]
            if dataframe_vars:
                context_parts.append(f"\nAVAILABLE DATAFRAMES IN MEMORY:")
                for var_name in dataframe_vars:
                    context_parts.append(f"- {var_name}: {available_vars[var_name]}")
                context_parts.append("\nPRIORITY: Use existing dataframes in memory rather than reloading from files when possible.")
    
    if completed_phases:
        context_parts.append("\nPREVIOUS WORK COMPLETED:")
        for phase, result in completed_phases.items():
            context_parts.append(f"\n{phase.upper()}:")
            context_parts.append(f"{result}")
            context_parts.append("-" * 50)
    
    context_parts.extend([
        f"\nYOUR CURRENT SPECIALIZATION FOCUS: {current_task}",
        "\nYou have access to all the same information a single comprehensive agent would have.",
        "Focus on your specialty area while being aware of the complete context and methodology.",
        "Build upon previous work without repeating it."
    ])
    
    return "\n".join(filter(None, context_parts))


# Native SDK specialist agent tools
@function_tool
async def call_business_understanding_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call business understanding specialist"""
    
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    result = await Runner.run(
        business_understanding_agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Track completion
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
    
    ctx.context.completed_phases["Business Understanding"] = str(result.final_output)
    
    # Track analytics
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call("Business Understanding Agent")
    
    return str(result.final_output)


@function_tool
async def call_data_understanding_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call data understanding specialist"""
    
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    result = await Runner.run(
        data_understanding_agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Track completion
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
        
    ctx.context.completed_phases["Data Understanding"] = str(result.final_output)
    
    # Track analytics
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call("Data Understanding Agent")
    
    return str(result.final_output)


@function_tool
async def call_data_preparation_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call data preparation specialist"""
    
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    result = await Runner.run(
        data_preparation_agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Track completion
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
        
    ctx.context.completed_phases["Data Preparation"] = str(result.final_output)
    
    # Track analytics
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call("Data Preparation Agent")
    
    return str(result.final_output)


@function_tool
async def call_modeling_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call modeling specialist"""
    
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    result = await Runner.run(
        modeling_agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Track completion
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
        
    ctx.context.completed_phases["Modeling"] = str(result.final_output)
    
    # Track analytics
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call("Modeling Agent")
    
    return str(result.final_output)


@function_tool
async def call_evaluation_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call evaluation specialist"""
    
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    result = await Runner.run(
        evaluation_agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Track completion
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
        
    ctx.context.completed_phases["Evaluation"] = str(result.final_output)
    
    # Track analytics
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call("Evaluation Agent")
    
    return str(result.final_output)


@function_tool
async def call_deployment_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call deployment specialist"""
    
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    result = await Runner.run(
        deployment_agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Track completion
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
        
    ctx.context.completed_phases["Deployment"] = str(result.final_output)
    
    # Track analytics
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call("Deployment Agent")
    
    return str(result.final_output)


async def run_multi_agent_analysis(prompt: str, file_name: str, max_turns: int = MAX_TURNS, model: str = DEFAULT_MODEL) -> AsyncGenerator[StreamingEvent, None]:
    """
    Run multi-agent analysis with simple, clean output display.
    
    Focus: Show specialist work clearly, then final summary.
    """
    
    # Reset execution environment
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
    context.analytics = analytics
    context.completed_phases = {}
    analytics.start_agent("Orchestrator")
    
    # Yield initial status
    yield StreamingEvent(
        event_type="analysis_start",
        content="🚀 Starting multi-agent analysis...",
        timestamp=context.start_time
    )
    
    try:
        with trace("Multi-Agent Data Science Analysis"):
            # Create model settings
            model_settings = ModelSettings(
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )

            # Create specialized agents - globals for tool functions to use
            global business_understanding_agent, data_understanding_agent, data_preparation_agent
            global modeling_agent, evaluation_agent, deployment_agent
            
            business_understanding_agent = Agent(
                name="Business Understanding Agent",
                model=model,
                model_settings=model_settings,
                instructions=BUSINESS_UNDERSTANDING_ENHANCED,
                tools=[execute_code],
            )

            data_understanding_agent = Agent(
                name="Data Understanding Agent", 
                model=model,
                model_settings=model_settings,
                instructions=DATA_UNDERSTANDING_ENHANCED,
                tools=[execute_code],
            )

            data_preparation_agent = Agent(
                name="Data Preparation Agent",
                model=model,
                model_settings=model_settings,
                instructions=DATA_PREPARATION_ENHANCED,
                tools=[execute_code],
            )

            modeling_agent = Agent(
                name="Modeling Agent",
                model=model,
                model_settings=model_settings,
                instructions=MODELING_ENHANCED,
                tools=[execute_code],
            )

            evaluation_agent = Agent(
                name="Evaluation Agent",
                model=model,
                model_settings=model_settings,
                instructions=EVALUATION_ENHANCED,
                tools=[execute_code],
            )

            deployment_agent = Agent(
                name="Deployment Agent",
                model=model,
                model_settings=model_settings,
                instructions=DEPLOYMENT_ENHANCED,
                tools=[execute_code],
            )

            # Create orchestrator agent
            orchestration_agent = Agent(
                name="Data Science Orchestration Agent",
                model=model,
                model_settings=model_settings,
                instructions=ORCHESTRATOR_ENHANCED,
                tools=[
                    call_business_understanding_agent,
                    call_data_understanding_agent,
                    call_data_preparation_agent,
                    call_modeling_agent,
                    call_evaluation_agent,
                    call_deployment_agent
                ]
            )

            # Run orchestrator with simple streaming
            result = Runner.run_streamed(
                orchestration_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Simple streaming - just orchestrator thinking and tool calls
            async for event in result.stream_events():
                # Check for cancellation
                if getattr(st.session_state, 'cancel_analysis', False):
                    result.cancel()
                    yield StreamingEvent(
                        event_type="analysis_cancelled",
                        content="🛑 Analysis cancelled by user",
                        timestamp=time.time(),
                        agent_name="Orchestrator"
                    )
                    return
                
                current_time = time.time()
                
                # Stream only key events
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Estimate tokens for cost tracking
                    analytics.estimate_tokens_from_content(event.data.delta)
                    
                    yield StreamingEvent(
                        event_type="text_delta",
                        content=event.data.delta,
                        timestamp=current_time,
                        agent_name="Orchestrator"
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Track tool calls
                        analytics.add_tool_call("Orchestrator")
                        
                        # Get tool name
                        tool_name = 'unknown_tool'
                        if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'function'):
                            tool_name = getattr(event.item.raw_item.function, 'name', 'unknown_tool')
                        
                        if 'agent' in tool_name:
                            agent_display_name = tool_name.replace('call_', '').replace('_agent', '').replace('_', ' ').title()
                            yield StreamingEvent(
                                event_type="tool_call",
                                content=f"🤖 Starting {agent_display_name} Agent...",
                                timestamp=current_time,
                                agent_name="Orchestrator"
                            )
                    
                    elif event.item.type == "tool_call_output_item":
                        # Get the tool output content
                        output_content = getattr(event.item, 'output', '')
                        
                        if output_content and len(output_content) > 50:
                            # Show brief summary
                            summary = output_content[:80] + "..." if len(output_content) > 80 else output_content
                            yield StreamingEvent(
                                event_type="tool_output",
                                content=f"✅ Completed: {summary}",
                                timestamp=current_time,
                                agent_name="Orchestrator"
                            )

            # Finish analytics
            analytics.finish_agent("Orchestrator")
            analytics.finish()

            # Get images for analytics
            images = get_created_images()
            for img in images:
                analytics.add_image(img)

            # Show final summary
            total_duration = time.time() - context.start_time
            
            yield StreamingEvent(
                event_type="analysis_complete",
                content=f"✅ Multi-agent analysis completed in {total_duration:.1f}s!\n\n📊 **Final Results:**\n{result.final_output}\n\n📸 Created {len(images)} visualizations",
                timestamp=time.time(),
                agent_name="Orchestrator"
            )
            
            # Now show specialist agent work clearly
            if hasattr(context, 'completed_phases') and context.completed_phases:
                yield StreamingEvent(
                    event_type="text_delta",
                    content="\n\n---\n\n## 🔍 **Specialist Agent Work:**\n\n",
                    timestamp=time.time(),
                    agent_name="System"
                )
                
                agent_display = {
                    "Business Understanding": "🎯 Business Understanding Agent",
                    "Data Understanding": "📊 Data Understanding Agent", 
                    "Data Preparation": "🔧 Data Preparation Agent",
                    "Modeling": "🤖 Modeling Agent",
                    "Evaluation": "📈 Evaluation Agent", 
                    "Deployment": "🚀 Deployment Agent"
                }
                
                for phase_name, phase_output in context.completed_phases.items():
                    if phase_output and phase_output.strip():
                        display_name = agent_display.get(phase_name, f"🔹 {phase_name}")
                        
                        yield StreamingEvent(
                            event_type="text_delta",
                            content=f"\n### {display_name}:\n{phase_output.strip()}\n\n",
                            timestamp=time.time(),
                            agent_name=phase_name
                        )
            
            # Analytics summary
            analytics_summary = analytics.get_summary()
            yield StreamingEvent(
                event_type="analytics_complete",
                content=f"\n---\n\n📊 **Analytics:** {analytics_summary['total_duration']:.1f}s, {analytics_summary['tool_calls']} tools, {analytics_summary['phases_completed']} phases, ${analytics_summary['estimated_cost']:.4f}",
                timestamp=time.time(),
                agent_name="System"
            )
            
    except Exception as e:
        # Finish analytics even on error
        analytics.finish_agent("Orchestrator")
        analytics.finish()
        
        yield StreamingEvent(
            event_type="analysis_error",
            content=f"❌ Multi-agent analysis failed: {str(e)}",
            timestamp=time.time(),
            agent_name="Orchestrator"
        )
    finally:
        # Ensure analytics is always finished
        if analytics and not analytics.end_time:
            analytics.finish()