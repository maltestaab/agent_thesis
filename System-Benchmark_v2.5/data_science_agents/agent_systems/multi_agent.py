"""
data_science_agents/agent_systems/multi_agent.py - Simplified multi-agent system

This module implements the multi-agent approach to data science analysis where
specialized agents handle different phases of the CRISP-DM methodology:

1. Business Understanding Agent - Defines objectives and success criteria
2. Data Understanding Agent - Explores and analyzes the dataset  
3. Data Preparation Agent - Cleans and prepares data for modeling
4. Modeling Agent - Builds and trains machine learning models
5. Evaluation Agent - Assesses results and provides business insights
6. Deployment Agent - Plans implementation and monitoring strategies

The orchestrator agent coordinates these specialists, calling them as needed
based on the analysis requirements and ensuring context flows between phases.
"""

import time
import streamlit as st
import asyncio
from typing import AsyncGenerator, Optional
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

# Core system imports
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images
from data_science_agents.core.context import AnalysisContext
from data_science_agents.core.analytics import setup_analytics, analytics_context, create_analytics_summary_event
from data_science_agents.core.events import (
    create_analysis_start_event, create_analysis_complete_event, 
    create_error_event, create_cancellation_event, create_event
)

# Configuration imports
from data_science_agents.config.settings import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TURNS, MAX_TURNS_SPECIALIST
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
    
    All specialist agents share the same basic setup:
    - They can execute Python code (execute_code tool)
    - They use the same model settings (temperature, top_p, etc.)
    - They follow the same configuration pattern
    
    Args:
        name: Human-readable agent name (e.g., "Data Understanding Agent")
        instructions: The specialized prompt/instructions for this agent's role
        model: AI model to use (gpt-4o-mini, gpt-4o, etc.)
        model_settings: Temperature, top_p, and other model configuration
        
    Returns:
        Agent configured and ready for specialist work
        
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
    
    This is a critical function that ensures each specialist agent has access
    to ALL the information a single comprehensive agent would have. It prevents
    specialists from being "blind" to work done by previous phases.
    
    Context includes:
    1. Original user request (so agents remember the goal)
    2. Available dataframes in memory (to avoid reloading data)  
    3. Results from all completed phases (to build upon previous work)
    4. Current specialization focus (what this agent should focus on)
    
    Args:
        user_request: The original analysis request from the user
        completed_phases: Dict of {phase_name: results} from previous agents
        current_task: Description of what this specialist should focus on
        data_file: Name of the dataset file being analyzed
        context: AnalysisContext with state information
        
    Returns:
        Formatted context string that gets prepended to the specialist's prompt
        
    Design Note: This ensures multi-agent mode has information parity with
    single-agent mode, just distributed across multiple specialists.
    """
    context_parts = [
        f"ANALYSIS REQUEST: {user_request}",
        f"ORIGINAL DATASET: {data_file}" if data_file else ""
    ]
    
    # Add available dataframes information to prevent redundant data loading
    if context:
        from data_science_agents.core.execution import get_available_variables
        available_vars = get_available_variables()
        if available_vars:
            dataframe_vars = [name for name, desc in available_vars.items() 
                            if 'DataFrame' in desc or 'pandas' in desc.lower()]
            if dataframe_vars:
                context_parts.append(f"\nAVAILABLE DATAFRAMES IN MEMORY:")
                for var_name in dataframe_vars:
                    context_parts.append(f"- {var_name}: {available_vars[var_name]}")
                context_parts.append("\nPRIORITY: Use existing dataframes in memory rather than reloading from files when possible.")
    
    # Add results from all previous phases so agents can build upon each other
    if completed_phases:
        context_parts.append("\nPREVIOUS WORK COMPLETED:")
        for phase, result in completed_phases.items():
            context_parts.append(f"\n{phase.upper()}:")
            context_parts.append(f"{result}")
            context_parts.append("-" * 50)
    
    # Define this specialist's focus area
    context_parts.extend([
        f"\nYOUR CURRENT SPECIALIZATION FOCUS: {current_task}",
        "\nYou have access to all the same information a single comprehensive agent would have.",
        "Focus on your specialty area while being aware of the complete context and methodology.",
        "Build upon previous work without repeating it."
    ])
    
    return "\n".join(filter(None, context_parts))


# =============================================================================
# GENERIC SPECIALIST CALLER
# =============================================================================
# This eliminates the massive duplication of 6x nearly identical function_tool
# functions. The original code had call_business_understanding_agent, 
# call_data_understanding_agent, etc. that all followed the exact same pattern
# with only the agent and phase name different.

async def call_specialist_agent(
    ctx: RunContextWrapper, 
    agent: Agent, 
    phase_name: str, 
    task_description: str
) -> str:
    """
    Generic function to call any specialist agent with standardized handling.
    
    This eliminates 150+ lines of duplicate code. The original system had
    6 separate function_tool functions that all did exactly the same thing:
    1. Create context for the specialist
    2. Call the specialist agent
    3. Store the results
    4. Track analytics
    5. Return the output
    
    Now all 6 specialists use this single, well-tested function.
    
    Args:
        ctx: Run context wrapper containing analysis state
        agent: The specialist agent to call
        phase_name: Name of the CRISP-DM phase (for result tracking)
        task_description: What this specialist should focus on
        
    Returns:
        String output from the specialist agent
        
    Process Flow:
    1. Create comprehensive context (includes previous work, available data, etc.)
    2. Call the specialist agent with max turns limit
    3. Store results in context for next phases
    4. Track analytics (tool calls, timing)
    5. Return results for orchestrator
    """
    # Create comprehensive context so specialist has full information
    specialist_context = create_full_context_for_specialist(
        user_request=ctx.context.original_prompt,
        completed_phases=getattr(ctx.context, 'completed_phases', {}),
        current_task=task_description,
        data_file=ctx.context.file_name,
        context=ctx.context
    )
    
    # Call the specialist agent with context and turn limits
    result = await Runner.run(
        agent,
        input=specialist_context,
        max_turns=MAX_TURNS_SPECIALIST,
        context=ctx.context
    )
    
    # Store results for future phases (ensure completed_phases exists)
    if not hasattr(ctx.context, 'completed_phases'):
        ctx.context.completed_phases = {}
    ctx.context.completed_phases[phase_name] = str(result.final_output)
    
    # Track analytics for monitoring
    if hasattr(ctx.context, 'analytics'):
        ctx.context.analytics.add_tool_call(f"{phase_name} Agent")
    
    return str(result.final_output)


# =============================================================================
# FUNCTION TOOLS FOR EACH SPECIALIST
# =============================================================================
# These are now greatly simplified using the generic specialist caller.
# Each function is now just 3-4 lines instead of 15-20 lines.

# Global variables to store agents (used by function tools)
# These are set during agent creation in run_multi_agent_analysis
_business_understanding_agent = None
_data_understanding_agent = None  
_data_preparation_agent = None
_modeling_agent = None
_evaluation_agent = None
_deployment_agent = None

@function_tool
async def call_business_understanding_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call business understanding specialist agent."""
    return await call_specialist_agent(
        ctx, _business_understanding_agent, "Business Understanding", task_description
    )

@function_tool  
async def call_data_understanding_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call data understanding specialist agent."""
    return await call_specialist_agent(
        ctx, _data_understanding_agent, "Data Understanding", task_description
    )

@function_tool
async def call_data_preparation_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call data preparation specialist agent."""
    return await call_specialist_agent(
        ctx, _data_preparation_agent, "Data Preparation", task_description
    )

@function_tool
async def call_modeling_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call modeling specialist agent."""
    return await call_specialist_agent(
        ctx, _modeling_agent, "Modeling", task_description
    )

@function_tool
async def call_evaluation_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call evaluation specialist agent."""
    return await call_specialist_agent(
        ctx, _evaluation_agent, "Evaluation", task_description
    )

@function_tool
async def call_deployment_agent(ctx: RunContextWrapper, task_description: str) -> str:
    """Call deployment specialist agent."""
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
    Run comprehensive data science analysis using multiple specialized agents.
    
    This is the main entry point for multi-agent analysis. It coordinates
    multiple specialist agents to complete a full data science workflow
    following the CRISP-DM methodology.
    
    Architecture:
    1. Orchestrator Agent - Manages the overall workflow and coordinates specialists
    2. Specialist Agents - Handle specific phases (Data Understanding, Modeling, etc.)
    3. Context Sharing - Ensures all agents have access to previous work
    4. Analytics Tracking - Monitors performance and costs across all agents
    5. Streaming Updates - Provides real-time progress to the UI
    
    Args:
        prompt: User's analysis request and requirements
        file_name: Dataset file to analyze
        max_turns: Maximum conversation turns for orchestrator (specialists have separate limits)
        model: AI model to use for all agents
        
    Yields:
        StreamingEvent objects for real-time UI updates during analysis
        
    Process Flow:
    1. Initialize environment and context
    2. Set up analytics tracking  
    3. Create all specialist agents using factory
    4. Create orchestrator agent with access to all specialists
    5. Run orchestrator with streaming updates
    6. Handle completion, errors, and cleanup
    """
    
    # === INITIALIZATION ===
    # Reset execution environment for clean start
    reset_execution_state()
    
    # Create analysis context to track state across agents
    context = AnalysisContext(
        file_name=file_name,
        analysis_type="multi_agent", 
        start_time=time.time(),
        original_prompt=prompt
    )
    
    # Initialize analytics with model-aware cost tracking
    analytics = setup_analytics(context, "Orchestrator", model)
    context.completed_phases = {}
    
    # Announce start of analysis
    yield create_analysis_start_event("multi_agent")
    
    try:
        with trace("Multi-Agent Data Science Analysis"):
            # === AGENT CREATION ===
            # Create model settings for all agents
            model_settings = ModelSettings(
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )

            # Create all specialist agents using factory (eliminates duplication)
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
            
            # Stream events to UI with simplified processing
            event_counter = 0
            async for event in result.stream_events():
                # Check for user cancellation
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
                
                # Handle different event types for UI updates
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Token-by-token streaming (like ChatGPT)
                    analytics.estimate_tokens_from_content(event.data.delta)
                    
                    yield create_event(
                        event_type="text_delta",
                        content=event.data.delta,
                        agent_name="Orchestrator"
                    )
                    
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Track tool usage
                        analytics.add_tool_call("Orchestrator")
                        
                        # Determine which specialist is being called
                        tool_name = 'unknown_tool'
                        if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'function'):
                            tool_name = getattr(event.item.raw_item.function, 'name', 'unknown_tool')
                        
                        if 'agent' in tool_name:
                            agent_display_name = tool_name.replace('call_', '').replace('_agent', '').replace('_', ' ').title()
                            yield create_event(
                                event_type="tool_call",
                                content=f"🤖 Starting {agent_display_name} Agent...",
                                agent_name="Orchestrator"
                            )
                    
                    elif event.item.type == "tool_call_output_item":
                        # Show brief summary of specialist work
                        output_content = getattr(event.item, 'output', '')
                        
                        if output_content and len(output_content) > 50:
                            summary = output_content[:80] + "..." if len(output_content) > 80 else output_content
                            yield create_event(
                                event_type="tool_output",
                                content=f"✅ Completed: {summary}",
                                agent_name="Orchestrator"
                            )

            # === COMPLETION HANDLING ===
            # Finish analytics tracking
            analytics.finish_agent("Orchestrator")
            analytics.finish()

            # Get final results and images
            images = get_created_images()
            for img in images:
                # Note: We don't track images in analytics anymore - execution.py is single source
                pass
            
            total_duration = time.time() - context.start_time

            # Show final comprehensive results
            final_content = f"✅ Multi-agent analysis completed in {total_duration:.1f}s!\n\n📊 **Final Results:**\n{result.final_output}"
            if len(images) > 0:
                final_content += f"\n\n📸 Created {len(images)} visualizations"
                
            yield create_event(
                event_type="analysis_complete",
                content=final_content,
                agent_name="Orchestrator"
            )
            
            # Show detailed specialist work if available
            if hasattr(context, 'completed_phases') and context.completed_phases:
                yield create_event(
                    event_type="text_delta",
                    content="\n\n---\n\n## 🔍 **Specialist Agent Work:**\n\n",
                    agent_name="System"
                )
                
                # Icons for each specialist type
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
        # Ensure analytics cleanup on error  
        analytics.finish_agent("Orchestrator")
        analytics.finish()
        
        yield create_error_event(f"Multi-agent analysis failed: {str(e)}")
        
    finally:
        # Guarantee analytics is always finished
        if analytics and not analytics.end_time:
            analytics.finish()