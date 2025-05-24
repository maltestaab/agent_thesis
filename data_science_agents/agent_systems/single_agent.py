"""
Single agent implementation for data science tasks
"""
import time
from agents import Agent, Runner, ModelSettings
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images, state
from data_science_agents.core.analytics import single_analytics
from data_science_agents.config.prompts import SINGLE_AGENT_INSTRUCTIONS, CORE_INSTRUCTION
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TURNS, MAX_TOKENS

# Create the single data science agent with enhanced context building
data_science_agent = Agent(
    name="Data Science Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=SINGLE_AGENT_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
    hooks=single_analytics  # Add analytics hooks
)

# Function to run the agent with enhanced output tracking and token limits
async def run_single_agent_analysis(prompt, max_turns=MAX_TURNS, max_tokens=MAX_TOKENS):
    """
    Run the data science agent with a specific prompt and enhanced output tracking.
    """
    reset_execution_state()
    
    print(f"🚀 Running Single Agent System: {prompt[:100]}...")
    print(f"🎛️ Token limit: {max_tokens:,}")
    
    result = await Runner.run(
        data_science_agent,
        prompt,
        max_turns=max_turns
    )
    
    # Estimate steps from result
    steps_count = len(result.messages) if hasattr(result, 'messages') else 1
    
    # Get analytics data with detailed metrics
    analytics_data = single_analytics.get_analytics_data(
        steps_count=steps_count,
        agent_calls=1  # Single agent always has 1 agent call
    )
    
    # Add created images to analytics
    analytics_data.created_images = get_created_images()
    
    # Print summary
    print(f"📊 Analysis completed in {analytics_data.duration:.2f}s")
    print(f"📈 {steps_count} steps • {len(analytics_data.tool_usage)} tool calls")
    print(f"📸 Created {len(analytics_data.created_images)} images")
    if analytics_data.errors:
        print(f"⚠️ Encountered {len(analytics_data.errors)} errors")
    
    return result, analytics_data