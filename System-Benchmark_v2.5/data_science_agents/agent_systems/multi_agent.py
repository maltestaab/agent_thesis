"""
data_science_agents/agent_systems/multi_agent.py - Multi-Agent Analysis System

This module implements a multi-agent approach to data science analysis where
specialized AI agents handle different phases of the CRISP-DM methodology:

1. Business Understanding Agent: Defines objectives and success criteria
2. Data Understanding Agent: Explores and analyzes the dataset structure
3. Data Preparation Agent: Cleans and prepares data for modeling
4. Modeling Agent: Builds and trains machine learning models
5. Evaluation Agent: Assesses results and provides business insights
6. Deployment Agent: Plans implementation and monitoring strategies

Architecture Features:
- Orchestrator Agent: Coordinates the workflow and manages specialist agents
- Specialist Agents: Each focuses on their domain expertise
- Context Sharing: Ensures all agents have access to previous work
- Real-time Streaming: Complete visibility into each specialist's work
- Analytics Tracking: Monitors performance and costs across all agents

The multi-agent approach offers:
- Specialized expertise for each analysis phase
- Parallel processing capabilities for complex analyses
- Detailed transparency into each phase of work
- Scalable architecture for adding new specialists
"""
import nest_asyncio
import logfire

nest_asyncio.apply()

logfire.configure(
    service_name='your-multi-agent-service',
    send_to_logfire=False,
)
logfire.instrument_openai_agents()

import time
import streamlit as st
import asyncio
from typing import AsyncGenerator
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

# Import core system components
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext
from data_science_agents.core.analytics import setup_analytics, create_analytics_summary_event
from data_science_agents.core.events import (
    create_analysis_start_event, create_error_event, create_cancellation_event, create_event
)

# Import configuration settings
from data_science_agents.config.settings import (
    DEFAULT_MODEL, MAX_TURNS, MAX_TURNS_SPECIALIST, create_model_settings
)
from data_science_agents.config.prompts import (
    BUSINESS_UNDERSTANDING_ENHANCED, DATA_UNDERSTANDING_ENHANCED, 
    DATA_PREPARATION_ENHANCED, MODELING_ENHANCED, EVALUATION_ENHANCED, 
    DEPLOYMENT_ENHANCED, ORCHESTRATOR_ENHANCED
)


# =============================================================================
# AGENT CREATION FACTORY
# =============================================================================

def create_specialist_agent(name: str, instructions: str, model: str, model_settings: ModelSettings) -> Agent:
    """
    Factory function for creating specialist agents with standardized configuration.
    
    This factory eliminates code duplication by providing a single function to create
    all specialist agents. Each specialist shares the same basic setup but with
    different names and specialized instructions for their domain expertise.
    
    All specialist agents have:
    - Python code execution capability (execute_code tool)
    - Consistent model settings (temperature, top_p, etc.)
    - Standardized configuration pattern
    
    Args:
        name (str): Human-readable agent name (e.g., "Data Understanding Agent")
        instructions (str): Specialized prompts for this agent's domain
        model (str): AI model to use (gpt-4o-mini, gpt-4o, etc.)
        model_settings (ModelSettings): Temperature, top_p, and other model config
        
    Returns:
        Agent: Configured specialist agent ready for analysis work
    """
    return Agent(
        name=name,
        model=model,
        model_settings=model_settings,
        instructions=instructions,
        tools=[execute_code],  # All specialists can execute Python code
    )


# =============================================================================
# CONTEXT CREATION HELPER  
# =============================================================================

def create_full_context_for_specialist(
    user_request: str, 
    completed_phases: dict, 
    current_task: str, 
    data_file: str = None, 
    context: AnalysisContext = None
) -> str:
    """
    Create comprehensive context information for specialist agents.
    
    This function ensures each specialist agent has access to all the information
    that a single comprehensive agent would have. It prevents specialists from
    being "blind" to work done by previous phases and enables them to build
    upon each other's results effectively.
    
    Context Components:
    1. Original user request: So agents remember the analysis goal
    2. Available dataframes: To avoid reloading data unnecessarily
    3. Previous phase results: To build upon completed work
    4. Current specialization focus: What this specific agent should focus on
    
    Args:
        user_request (str): The original analysis request from the user
        completed_phases (dict): Results from all previously completed phases
        current_task (str): Description of what this specialist should focus on
        data_file (str): Name of the dataset file being analyzed
        context (AnalysisContext): Current analysis context with state information
        
    Returns:
        str: Formatted context string for the specialist agent
    """
    context_parts = [
        f"ANALYSIS REQUEST: {user_request}",
        f"ORIGINAL DATASET: {data_file}" if data_file else ""
    ]
    
    # Add information about available dataframes to prevent redundant loading
    if context:
        from data_science_agents.core.execution import get_available_variables
        available_vars = get_available_variables()
        if available_vars:
            # Identify dataframe variables specifically
            dataframe_vars = [name for name, desc in available_vars.items() 
                            if 'DataFrame' in desc or 'pandas' in desc.lower()]
            if dataframe_vars:
                context_parts.append(f"\nAVAILABLE DATAFRAMES IN MEMORY:")
                for var_name in dataframe_vars:
                    context_parts.append(f"- {var_name}: {available_vars[var_name]}")
                context_parts.append("\nPRIORITY: Use existing dataframes in memory rather than reloading from files when possible.")
    
    # Include results from all previous phases for continuity
    if completed_phases:
        context_parts.append("\nPREVIOUS WORK COMPLETED:")
        for phase, result in completed_phases.items():
            context_parts.append(f"\n{phase.upper()}:")
            context_parts.append(f"{result}")
            context_parts.append("-" * 50)
    
    # Define this specialist's specific focus area
    context_parts.extend([
        f"\nYOUR CURRENT SPECIALIZATION FOCUS: {current_task}",
        "\nYou have access to all the same information a single comprehensive agent would have.",
        "Focus on your specialty area while being aware of the complete context and methodology.",
        "Build upon previous work without repeating it."
    ])
    
    return "\n".join(filter(None, context_parts))


# =============================================================================
# STREAMING SPECIALIST CALLER
# =============================================================================

async def call_specialist_agent(
    ctx: RunContextWrapper, 
    agent: Agent, 
    phase_name: str, 
    task_description: str
) -> str:
    """
    Generic function to call any specialist agent with full streaming and analytics.
    
    This function provides a unified interface for calling all specialist agents.
    It uses streaming to capture real-time work from specialists, providing complete
    transparency into their reasoning, code execution, and results.
    
    Key Features:
    - Real-time streaming of specialist work
    - Exact token counts and tool calls (no estimation needed)
    - Complete transparency into specialist reasoning
    - Automatic analytics tracking and context sharing
    
    Args:
        ctx (RunContextWrapper): Analysis context containing current state
        agent (Agent): The specialist agent to execute
        phase_name (str): Name of the CRISP-DM phase for tracking
        task_description (str): Specific task for this specialist to focus on
        
    Returns:
        str: Complete output from the specialist agent
    """
    from data_science_agents.core.events import create_event
    from openai.types.responses import ResponseTextDeltaEvent
    
    # Create comprehensive context for the specialist
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    # Initialize specialist events collection for main stream
    if not hasattr(ctx.context, 'specialist_events'):
        ctx.context.specialist_events = []
    
    # Announce which specialist is starting work
    ctx.context.specialist_events.append(create_event(
        event_type="specialist_start",
        content=f"🤖 Starting {phase_name} Agent...",
        agent_name="Orchestrator"
    ))
    
    # === EXECUTE SPECIALIST WITH STREAMING ===
    # Use streaming to capture real-time work from the specialist
    result = Runner.run_streamed(
        agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Process specialist's streaming events
    specialist_output = ""
    specialist_event_count = 0
    
    async for event in result.stream_events():
        specialist_event_count += 1
        
        # === PROCESS SPECIALIST STREAMING EVENTS ===
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            # Capture token-by-token streaming from specialist
            delta_content = event.data.delta
            specialist_output += delta_content
            
            # Store specialist text events for main stream to display
            ctx.context.specialist_events.append(create_event(
                event_type="text_delta",
                content=delta_content,
                agent_name=f"{phase_name} Agent"
            ))
            
            # Track tokens for analytics (automatic cost calculation)
            if hasattr(ctx.context, 'analytics'):
                ctx.context.analytics.estimate_tokens_from_content(delta_content)
                
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                # Specialist is executing code
                if hasattr(ctx.context, 'analytics'):
                    ctx.context.analytics.add_tool_call(f"{phase_name} Agent")
                
                ctx.context.specialist_events.append(create_event(
                    event_type="tool_call", 
                    content=f"🔧 {phase_name} Agent executing code...",
                    agent_name=f"{phase_name} Agent"
                ))
                
            elif event.item.type == "tool_call_output_item":
                # Specialist's code execution completed
                output_text = str(event.item.output)
                preview = output_text[:100] + "..." if len(output_text) > 100 else output_text
                
                ctx.context.specialist_events.append(create_event(
                    event_type="tool_output",
                    content=f"✅ {phase_name} Agent: {preview}",
                    agent_name=f"{phase_name} Agent"
                ))
    
    # Get final specialist output
    final_specialist_output = str(result.final_output)
    if not final_specialist_output:
        final_specialist_output = specialist_output
    
    # Store results for future phases to build upon
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
    ctx.context.completed_phases[phase_name] = final_specialist_output
    
    # Announce specialist completion
    ctx.context.specialist_events.append(create_event(
        event_type="specialist_complete",
        content=f"✅ {phase_name} Agent completed ({specialist_event_count} events)",
        agent_name="Orchestrator"
    ))
    
    return final_specialist_output


# =============================================================================
# FUNCTION TOOLS FOR EACH SPECIALIST
# =============================================================================

# Global variables to store specialist agents (used by function tools)
# These are initialized during agent creation in run_multi_agent_analysis
_business_understanding_agent = None
_data_understanding_agent = None  
_data_preparation_agent = None
_modeling_agent = None
_evaluation_agent = None
_deployment_agent = None

@function_tool
async def call_business_understanding_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """
    Call the Business Understanding specialist agent.
    
    This agent focuses on defining business objectives, success criteria,
    and converting business problems into data science problems.
    """
    return await call_specialist_agent(
        ctx, _business_understanding_agent, "Business Understanding", task_description
    )

@function_tool  
async def call_data_understanding_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """
    Call the Data Understanding specialist agent.
    
    This agent focuses on data exploration, quality assessment, and
    discovering initial insights about the dataset structure and patterns.
    """
    return await call_specialist_agent(
        ctx, _data_understanding_agent, "Data Understanding", task_description
    )

@function_tool
async def call_data_preparation_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """
    Call the Data Preparation specialist agent.
    
    This agent focuses on data cleaning, feature engineering, and
    transforming data into formats suitable for modeling.
    """
    return await call_specialist_agent(
        ctx, _data_preparation_agent, "Data Preparation", task_description
    )

@function_tool
async def call_modeling_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """
    Call the Modeling specialist agent.
    
    This agent focuses on selecting appropriate algorithms, building models,
    and evaluating their technical performance.
    """
    return await call_specialist_agent(
        ctx, _modeling_agent, "Modeling", task_description
    )

@function_tool
async def call_evaluation_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """
    Call the Evaluation specialist agent.
    
    This agent focuses on assessing model results against business criteria
    and providing actionable insights and recommendations.
    """
    return await call_specialist_agent(
        ctx, _evaluation_agent, "Evaluation", task_description
    )

@function_tool
async def call_deployment_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """
    Call the Deployment specialist agent.
    
    This agent focuses on creating deployment strategies, monitoring plans,
    and implementation guidance for putting models into production.
    """
    return await call_specialist_agent(
        ctx, _deployment_agent, "Deployment", task_description
    )


# =============================================================================
# MAIN MULTI-AGENT ANALYSIS FUNCTION
# =============================================================================

async def run_multi_agent_analysis(
    prompt: str, 
    file_name: str, 
    max_turns: int = MAX_TURNS, 
    model: str = DEFAULT_MODEL
) -> AsyncGenerator:
    """
    Execute comprehensive data science analysis using multiple specialized agents.
    
    This function orchestrates a complete data science workflow using an orchestrator
    agent that coordinates multiple specialist agents. Each specialist focuses on
    their domain expertise while the orchestrator manages the overall workflow
    and ensures proper context sharing between phases.
    
    Architecture Components:
    1. Orchestrator Agent: Manages workflow and coordinates specialists
    2. Specialist Agents: Handle specific CRISP-DM phases
    3. Context Sharing: Ensures all agents have access to previous work
    4. Analytics Tracking: Monitors performance across all agents
    5. Streaming Updates: Provides real-time visibility into all agent work
    
    Args:
        prompt (str): User's analysis request and requirements
        file_name (str): Dataset file to analyze
        max_turns (int): Maximum conversation turns for orchestrator
        model (str): AI model to use for all agents
        
    Yields:
        StreamingEvent: Real-time events for UI updates during analysis
        
    Process Flow:
    1. Initialize environment and create analysis context
    2. Set up analytics tracking with model-specific costs
    3. Create all specialist agents using the factory function
    4. Create orchestrator agent with access to all specialists
    5. Execute orchestrator with real-time streaming
    6. Collect and stream specialist events for transparency
    7. Handle completion, errors, and cleanup
    """
    
    # === ENVIRONMENT INITIALIZATION ===
    # Reset execution environment for clean analysis start
    reset_execution_state()
    
    # Create analysis context to track state across all agents
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="multi_agent", 
        start_time=time.time(),
        original_prompt=prompt
    )
    
    # Initialize analytics tracking with model-aware cost calculation
    analytics = setup_analytics(context, "Orchestrator", model)
    context.completed_phases = {}
    
    # Notify user interface that multi-agent analysis is starting
    yield create_analysis_start_event("multi_agent")
    
    try:
        with trace("Multi-Agent Data Science Analysis"):
            # === AGENT CREATION ===
            # Create consistent model settings for all agents
            model_settings = model_settings = create_model_settings(model)


            # Create all specialist agents using the factory function
            global _business_understanding_agent, _data_understanding_agent
            global _data_preparation_agent, _modeling_agent, _evaluation_agent, _deployment_agent
            
            _business_understanding_agent = create_specialist_agent(
                "Business Understanding Agent", BUSINESS_UNDERSTANDING_ENHANCED, model, model_settings
            )
            _data_understanding_agent = create_specialist_agent(
                "Data Understanding Agent", DATA_UNDERSTANDING_ENHANCED, model, model_settings
            )
            _data_preparation_agent = create_specialist_agent(
                "Data Preparation Agent", DATA_PREPARATION_ENHANCED, model, model_settings
            )
            _modeling_agent = create_specialist_agent(
                "Modeling Agent", MODELING_ENHANCED, model, model_settings
            )
            _evaluation_agent = create_specialist_agent(
                "Evaluation Agent", EVALUATION_ENHANCED, model, model_settings
            )
            _deployment_agent = create_specialist_agent(
                "Deployment Agent", DEPLOYMENT_ENHANCED, model, model_settings
            )

            # Create orchestrator agent with access to all specialists
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

            # === ANALYSIS EXECUTION ===
            # Run orchestrator with streaming updates
            result = Runner.run_streamed(
                orchestration_agent,
                input=prompt,
                context=context,
                max_turns=max_turns
            )
            
            # Stream events to UI with specialist event processing
            event_counter = 0
            last_specialist_event_check = 0
            
            async for event in result.stream_events():
                # Check for user cancellation via stop button
                if getattr(st.session_state, 'cancel_analysis', False):
                    result.cancel()
                    yield create_cancellation_event()
                    return
                
                current_time = time.time()
                event_counter += 1
                
                # === YIELD SPECIALIST EVENTS ===
                # Check for accumulated specialist events and stream them
                if hasattr(context, 'specialist_events') and len(context.specialist_events) > last_specialist_event_check:
                    for i in range(last_specialist_event_check, len(context.specialist_events)):
                        yield context.specialist_events[i]
                    last_specialist_event_check = len(context.specialist_events)
                
                # === PERIODIC ANALYTICS UPDATES ===
                # Share analytics with UI every 10 events for live display
                if event_counter % 10 == 0 and analytics:
                    from data_science_agents.core.events import create_analytics_update_event
                    yield create_analytics_update_event(analytics)
                
                # Handle different event types for orchestrator UI updates
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Token-by-token streaming from orchestrator (like ChatGPT)
                    analytics.estimate_tokens_from_content(event.data.delta)
                    
                    yield create_event(
                        event_type="text_delta",
                        content=event.data.delta,
                        agent_name="Orchestrator"
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Track orchestrator tool usage (calling specialists)
                        analytics.add_tool_call("Orchestrator")
                        
                        # Determine which specialist is being called
                        tool_name = 'unknown_tool'
                        if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'function'):
                            tool_name = getattr(event.item.raw_item.function, 'name', 'unknown_tool')
                        
                        if 'agent' in tool_name:
                            # Format tool name for display
                            agent_display_name = tool_name.replace('call_', '').replace('_agent', '').replace('_', ' ').title()
                            yield create_event(
                                event_type="tool_call",
                                content=f"🤖 Starting {agent_display_name} Agent...",
                                agent_name="Orchestrator"
                            )
                    
                    elif event.item.type == "tool_call_output_item":
                        # Show brief summary of specialist work completion
                        output_content = getattr(event.item, 'output', '')
                        
                        if output_content and len(output_content) > 50:
                            summary = output_content[:80] + "..." if len(output_content) > 80 else output_content
                            yield create_event(
                                event_type="tool_output",
                                content=f"✅ Completed: {summary}",
                                agent_name="Orchestrator"
                            )

            # === COMPLETION HANDLING ===
            # Finalize analytics tracking
            analytics.finish_agent("Orchestrator")
            analytics.finish()

            # Get final results and created visualizations
            images = get_created_images()
            total_duration = time.time() - context.start_time

            # Create comprehensive completion message
            final_content = f"✅ Multi-agent analysis completed in {total_duration:.1f}s!\n\n📊 **Final Results:**\n{result.final_output}"
            if len(images) > 0:
                final_content += f"\n\n📸 Created {len(images)} visualizations"
                
            yield create_event(
                event_type="analysis_complete",
                content=final_content,
                agent_name="Orchestrator"
            )
            
            # Display detailed specialist work if available
            if hasattr(context, 'completed_phases') and context.completed_phases:
                yield create_event(
                    event_type="text_delta",
                    content="\n\n---\n\n## 🔍 **Specialist Agent Work:**\n\n",
                    agent_name="System"
                )
                
                # Icons and display names for each specialist type
                agent_display = {
                    "Business Understanding": "🎯 Business Understanding Agent",
                    "Data Understanding": "📊 Data Understanding Agent", 
                    "Data Preparation": "🔧 Data Preparation Agent",
                    "Modeling": "🤖 Modeling Agent",
                    "Evaluation": "📈 Evaluation Agent", 
                    "Deployment": "🚀 Deployment Agent"
                }
                
                # Display work from each completed phase
                for phase_name, phase_output in context.completed_phases.items():
                    if phase_output and phase_output.strip():
                        display_name = agent_display.get(phase_name, f"🔹 {phase_name}")
                        
                        yield create_event(
                            event_type="text_delta",
                            content=f"\n### {display_name}:\n{phase_output.strip()}\n\n",
                            agent_name=phase_name
                        )
            
            # Final analytics summary
            analytics_summary = create_analytics_summary_event(analytics)
            yield create_event(
                event_type="analytics_complete",
                content=f"\n---\n\n{analytics_summary}",
                agent_name="System"
            )
            
    except Exception as e:
        # === ERROR HANDLING ===
        # Ensure analytics cleanup even on errors
        analytics.finish_agent("Orchestrator")
        analytics.finish()
        
        yield create_error_event(f"Multi-agent analysis failed: {str(e)}")
        
    finally:
        # === CLEANUP ===
        # Guarantee analytics is always properly finished
        if analytics and not analytics.end_time:
            analytics.finish()