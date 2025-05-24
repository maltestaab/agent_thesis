"""
Multi-agent system implementation for data science tasks
"""
import time
from agents import Agent, Runner, ModelSettings
from data_science_agents.core.execution import execute_code, reset_execution_state, get_created_images, state
from data_science_agents.core.analytics import multi_analytics
from data_science_agents.config.prompts import (
    ORCHESTRATOR_INSTRUCTIONS, 
    DATA_EXPLORATION_INSTRUCTIONS,
    DATA_PREPROCESSING_INSTRUCTIONS,
    FEATURE_SELECTION_INSTRUCTIONS,
    MODELING_INSTRUCTIONS,
    VISUALIZATION_INSTRUCTIONS,
    INSIGHTS_INSTRUCTIONS,
    CORE_INSTRUCTION
)
from data_science_agents.config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TURNS, MAX_TOKENS

# Custom output extractor to capture both text output and structured results
async def extract_structured_output(run_result):
    """Extract both the agent's text output and the 'result' variable from state"""
    output_data = {
        'text_output': run_result.final_output,
        'structured_result': None
    }
    
    # Try to get the 'result' variable from state
    if 'result' in state.namespace:
        output_data['structured_result'] = state.namespace['result']
    
    return output_data

# Define the specialized data science agents with clear boundaries

# 1. Data Exploration Agent - Phase 1 of the analysis
data_exploration_agent = Agent(
    name="Data Exploration Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=DATA_EXPLORATION_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
)

# 2. Data Preprocessing Agent - Phase 2 of the analysis  
data_preprocessing_agent = Agent(
    name="Data Preprocessing Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=DATA_PREPROCESSING_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
)

# 3. Feature Selection Agent - Phase 3 of the analysis
feature_selection_agent = Agent(
    name="Feature Selection Agent", 
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=FEATURE_SELECTION_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
)

# 4. Modeling Agent - Phase 4 of the analysis
modeling_agent = Agent(
    name="Modeling Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=MODELING_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
)

# 5. Visualization Agent - Phase 5 of the analysis
visualization_agent = Agent(
    name="Visualization Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=VISUALIZATION_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
)

# 6. Insights Agent - Phase 6 of the analysis
insights_agent = Agent(
    name="Insights Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=INSIGHTS_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[execute_code],
)

# Create the orchestration agent with enhanced context management
orchestration_agent = Agent(
    name="Data Science Orchestration Agent",
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.3
    ),
    instructions=ORCHESTRATOR_INSTRUCTIONS.format(core_instruction=CORE_INSTRUCTION),
    tools=[
        data_exploration_agent.as_tool(
            tool_name="data_exploration_agent",
            tool_description="ONLY handles initial data exploration: loads data, examines structure (shape/columns/types), checks missing values, basic statistics. Does NOT clean data, select features, or build models. Returns: data structure summary, missing value report, basic statistics.",
            custom_output_extractor=extract_structured_output
        ),
        data_preprocessing_agent.as_tool(
            tool_name="data_preprocessing_agent", 
            tool_description="ONLY handles data cleaning and preparation: missing value imputation, data type conversion, feature scaling/normalization, categorical encoding. Does NOT explore data or select features. Requires: data already explored. Returns: cleaned dataset, preprocessing summary.",
            custom_output_extractor=extract_structured_output
        ),
        feature_selection_agent.as_tool(
            tool_name="feature_selection_agent",
            tool_description="ONLY identifies important features: statistical analysis, correlation analysis, feature importance ranking. Does NOT build models or create visualizations. Requires: clean preprocessed data. Returns: ranked list of important features with justification.",
            custom_output_extractor=extract_structured_output
        ),
        modeling_agent.as_tool(
            tool_name="modeling_agent",
            tool_description="ONLY builds and evaluates statistical models: model training, performance evaluation, feature importance from models. Does NOT select features or create visualizations. Requires: processed data and selected features. Returns: model performance metrics, model-based feature importance.",
            custom_output_extractor=extract_structured_output
        ),
        visualization_agent.as_tool(
            tool_name="visualization_agent",
            tool_description="ONLY creates visualizations: plots for feature importance, model results, data distributions. Does NOT perform analysis or provide insights. Requires: analysis results from previous phases. Returns: saved visualization files and description of plots created.",
            custom_output_extractor=extract_structured_output
        ),
        insights_agent.as_tool(
            tool_name="insights_agent", 
            tool_description="ONLY synthesizes final insights: integrates all findings, provides business interpretation, creates final summary. Does NOT perform any analysis. Requires: results from all previous phases. Returns: comprehensive final insights and conclusions.",
            custom_output_extractor=extract_structured_output
        )
    ],
)

# Function to run the multi-agent system with enhanced features
async def run_multi_agent_analysis(prompt, max_turns=MAX_TURNS, max_tokens=MAX_TOKENS):
    """
    Run the multi-agent data science system with a specific prompt.
    """
    reset_execution_state()
    
    print(f"🚀 Running Multi-Agent System: {prompt[:100]}...")
    print(f"🎛️ Token limit: {max_tokens:,}")
    
    start_time = time.time()
    
    result = await Runner.run(
        orchestration_agent,
        prompt,
        max_turns=max_turns
    )
    duration = time.time() - start_time
    
    # Estimate steps and agents from result
    steps_count = len(result.messages) if hasattr(result, 'messages') else 6  # Default to 6 agents
    agent_calls = 6  # Multi-agent has 6 specialized agents
    
    # Record analytics with image count
    analytics_data = multi_analytics.record_run(duration, steps_count, agent_calls)
    analytics_data['created_images'] = get_created_images()
    
    print(f"📸 Created {len(analytics_data['created_images'])} images during analysis")
    
    return result, analytics_data