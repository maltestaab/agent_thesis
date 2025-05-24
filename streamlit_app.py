"""
Streamlit web interface for the data science agent comparison system
"""
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import pandas as pd

# Load environment variables
load_dotenv()

# Import functions from both agent systems
from data_science_agents.agent_systems.single_agent import run_single_agent_analysis
from data_science_agents.agent_systems.multi_agent import run_multi_agent_analysis
from data_science_agents.core.analytics import single_analytics, multi_analytics
from data_science_agents.utils.display_helpers import display_output_with_inline_images



# Global variable to track if analysis is running
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Set page configuration
st.set_page_config(
    page_title="Data Science Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("Data Science Analysis Tool")
st.markdown("Upload your data and specify your analysis needs - our AI will handle the rest!")

# Add agent mode selector in the sidebar
st.sidebar.title("Settings")
agent_mode = st.sidebar.radio(
    "Choose Agent System",
    ["Single Agent", "Multi-Agent"],
    help="Single Agent uses one LLM to handle the entire analysis. Multi-Agent uses specialized agents for different phases."
)

# Add task management controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Task Management")

# Show current task status
if st.session_state.analysis_running:
    st.sidebar.warning("🔄 Agent is currently running...")
    st.sidebar.info("Running analysis... Please wait.")
else:
    st.sidebar.info("✅ No active analysis")

# Token limit setting
st.sidebar.markdown("---")
st.sidebar.markdown("### Token Limits")
max_tokens = st.sidebar.number_input(
    "Max tokens per run",
    min_value=1000,
    max_value=1000000,
    value=500000,
    step=10000,
    help="Maximum tokens to prevent runaway costs"
)

# File upload section
st.subheader("Upload your data")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Show file information and preview if uploaded
if uploaded_file is not None:
    st.success(f"File uploaded: **{uploaded_file.name}**")
    
    # Show a preview of the data
    try:
        if uploaded_file.name.endswith('.csv'):
            preview_df = pd.read_csv(uploaded_file)
        else:
            try:
                preview_df = pd.read_excel(uploaded_file)
            except ImportError:
                st.warning("Excel files require 'openpyxl' library. Install with: `pip install openpyxl`")
                st.info("For now, please use CSV files or install openpyxl to view Excel files.")
                preview_df = None
        
        if preview_df is not None:
            st.subheader("Data Preview")
            st.dataframe(preview_df.head(), use_container_width=True)
            st.markdown(f"**Shape:** {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
            st.markdown(f"**Columns:** {', '.join(preview_df.columns.tolist())}")
            
            # Reset file pointer for later use
            uploaded_file.seek(0)
        
    except Exception as e:
        st.warning(f"Could not preview the file: {str(e)}")

# User prompt input
st.subheader("Enter your analysis request")
user_prompt = st.text_area(
    "What would you like to analyze?",
    "Please analyze this data and tell me which are the most important features using a statistical model. Visualize the results and provide the insights in a clear and structured way.",
    height=100
)

# Display the selected mode
st.subheader("Analysis Mode")
if agent_mode == "Single Agent":
    st.info("Using **Single Agent** mode: One agent will handle the entire analysis process.")
else:
    st.info("Using **Multi-Agent** mode: Specialized agents will collaborate to complete the analysis.")

# Add a run button with the selected mode
run_button_text = f"Run Analysis with {agent_mode}"

# Check if analysis is already running - ensure we always get a boolean
analysis_running = bool(st.session_state.analysis_running)

def display_analytics(analytics_data):
    """Display detailed analytics in Streamlit"""
    st.subheader("📊 Performance Analytics")
    
    # Overall metrics
    summary = analytics_data.get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Duration", f"{summary['total_duration']:.2f}s")
    with col2:
        st.metric("Tool Calls", summary['total_tool_calls'])
    with col3:
        st.metric("Error Rate", f"{summary['error_rate']*100:.1f}%")
    with col4:
        st.metric("Images Created", summary['total_images'])
    
    # Tool Usage Analysis
    st.subheader("🔧 Tool Usage")
    tool_data = []
    for tool_name, metrics in analytics_data.tool_metrics.items():
        tool_data.append({
            "Tool": tool_name,
            "Calls": metrics.total_calls,
            "Avg Duration": f"{metrics.average_duration:.2f}s",
            "Total Time": f"{metrics.total_duration:.2f}s",
            "Errors": metrics.error_count
        })
    if tool_data:
        st.dataframe(pd.DataFrame(tool_data))
    
    # Agent Performance (especially useful for multi-agent)
    st.subheader("🤖 Agent Performance")
    agent_data = []
    for agent_name, metrics in analytics_data.agent_metrics.items():
        agent_data.append({
            "Agent": agent_name,
            "Total Time": f"{metrics.total_time:.2f}s",
            "Thinking Time": f"{metrics.thinking_time:.2f}s",
            "Tool Time": f"{metrics.tool_time:.2f}s",
            "Tool Calls": metrics.tool_calls,
            "Errors": metrics.error_count
        })
    if agent_data:
        st.dataframe(pd.DataFrame(agent_data))
    
    # Timeline Visualization
    st.subheader("⏱️ Execution Timeline")
    # Create a timeline of tool usage
    if analytics_data.tool_usage:
        timeline_data = []
        start_time = min(u['start_time'] for u in analytics_data.tool_usage)
        for usage in analytics_data.tool_usage:
            timeline_data.append({
                'Tool': usage['tool'],
                'Agent': usage['agent'],
                'Start': usage['start_time'] - start_time,
                'Duration': usage.get('duration', 0),
                'Status': 'Error' if 'error' in usage else 'Success'
            })
        timeline_df = pd.DataFrame(timeline_data)
        
        # Plot timeline using Plotly
        import plotly.express as px
        fig = px.timeline(timeline_df, 
                         x_start='Start',
                         x_end=timeline_df['Start'] + timeline_df['Duration'],
                         y='Agent',
                         color='Tool',
                         hover_data=['Duration', 'Status'])
        fig.update_layout(title='Tool Usage Timeline')
        st.plotly_chart(fig)

if st.button(run_button_text, disabled=analysis_running):
    if uploaded_file is None:
        st.error("Please upload a file first.")
    else:
        # Set analysis as running
        st.session_state.analysis_running = True
        
        # Save the uploaded file with original filename
        file_name = uploaded_file.name
        local_path = file_name
        
        # Only save if file doesn't already exist
        if not os.path.exists(local_path):
            with open(local_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
        
        # Show a progress message
        with st.spinner(f"Running analysis with {agent_mode}... This may take a few minutes."):
            try:
                # Create an enhanced prompt with explicit dataset information
                enhanced_prompt = f"""
Dataset Information:
- File name: {file_name}
- This is the dataset you should analyze for this request

Analysis Request: {user_prompt}

Important Instructions:
- Use the file '{file_name}' for your analysis
- If you need to load the data, use: pd.read_csv('{file_name}') for CSV files or pd.read_excel('{file_name}') for Excel files
- The dataset is available in the current working directory
- Maximum tokens allowed: {max_tokens:,}
"""
                
                # Simplified async execution for Streamlit
                async def run_analysis():
                    if agent_mode == "Single Agent":
                        return await run_single_agent_analysis(enhanced_prompt, max_turns=50, max_tokens=max_tokens)
                    else:
                        return await run_multi_agent_analysis(enhanced_prompt, max_turns=50, max_tokens=max_tokens)
                
                # Execute the async function - handle event loop properly
                try:
                    # Check if there's already an event loop running
                    try:
                        loop = asyncio.get_running_loop()
                        # If we reach here, there's already a loop running
                        # Use nest_asyncio to allow nested loops
                        try:
                            import nest_asyncio
                            nest_asyncio.apply()
                            result, analytics_data = asyncio.run(run_analysis())
                        except ImportError:
                            st.error("Please install nest_asyncio: pip install nest_asyncio")
                            result, analytics_data = None, None
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run()
                        result, analytics_data = asyncio.run(run_analysis())
                except Exception as async_error:
                    st.error(f"Async execution error: {str(async_error)}")
                    result, analytics_data = None, None
                
                # Check if analysis was successful
                if result is not None and analytics_data is not None:
                    # Display the results with inline images
                    st.subheader("Analysis Results")
                    
                    # Get created images from analytics data
                    created_images = analytics_data.get('created_images', [])
                    
                    # Display final output with inline images
                    display_output_with_inline_images(result.final_output, created_images)
                    
                    # Display detailed analytics
                    display_analytics(analytics_data)
                    
                    # Display analytics information
                    if analytics_data:
                        image_count = len(created_images)
                        st.info(f"Completed in {analytics_data['duration']:.2f}s • "
                               f"{analytics_data['total_steps']} steps • "
                               f"{analytics_data['agent_calls']} agents • "
                               f"{analytics_data['tool_calls']} tools • "
                               f"{image_count} images created")
                    
                    # Display system info
                    st.subheader("System Information")
                    st.markdown(f"**Agent System:** {agent_mode}")
                    st.markdown(f"**Dataset:** {file_name}")
                    st.markdown(f"**Token Limit:** {max_tokens:,}")
                    if analytics_data:
                        st.markdown(f"**Execution Time:** {analytics_data['duration']:.2f} seconds")
                        st.markdown(f"**Total Steps:** {analytics_data['total_steps']}")
                        st.markdown(f"**Images Created:** {len(created_images)}")
                    
                    if agent_mode == "Multi-Agent":
                        st.markdown("**Process:** Used specialized agents for exploration, preprocessing, feature selection, modeling, visualization, and insights.")
                    else:
                        st.markdown("**Process:** Used a single agent for the entire analysis workflow.")
                else:
                    st.warning("Analysis completed with issues or was interrupted.")
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
            finally:
                # Reset analysis running state
                st.session_state.analysis_running = False

# Analytics History in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Performance Analytics")


if hasattr(single_analytics, 'history') and single_analytics.history:
    single_runs = [r for r in single_analytics.history if r.get('duration')]
    if single_runs:
        avg_single = sum(r['duration'] for r in single_runs) / len(single_runs)
        st.sidebar.metric("Single Agent Avg", f"{avg_single:.2f}s", 
                         delta=f"{len(single_runs)} runs")

if hasattr(multi_analytics, 'history') and multi_analytics.history:
    multi_runs = [r for r in multi_analytics.history if r.get('duration')]
    if multi_runs:
        avg_multi = sum(r['duration'] for r in multi_runs) / len(multi_runs)
        st.sidebar.metric("Multi-Agent Avg", f"{avg_multi:.2f}s", 
                         delta=f"{len(multi_runs)} runs")

# Show basic run information in sidebar
if agent_mode == "Single Agent" and hasattr(single_analytics, 'last_run') and single_analytics.last_run:
    st.sidebar.markdown("#### Last Single Agent Run")
    st.sidebar.markdown(f"Steps: {single_analytics.last_run.get('total_steps', 0)}")
    st.sidebar.markdown(f"Duration: {single_analytics.last_run.get('duration', 0):.2f}s")

if agent_mode == "Multi-Agent" and hasattr(multi_analytics, 'last_run') and multi_analytics.last_run:
    st.sidebar.markdown("#### Last Multi-Agent Run")
    st.sidebar.markdown(f"Steps: {multi_analytics.last_run.get('total_steps', 0)}")
    st.sidebar.markdown(f"Agents: {multi_analytics.last_run.get('agent_calls', 0)}")
    st.sidebar.markdown(f"Tools: {multi_analytics.last_run.get('tool_calls', 0)}")
    st.sidebar.markdown(f"Duration: {multi_analytics.last_run.get('duration', 0):.2f}s")

# Add comparison information
st.sidebar.markdown("---")
st.sidebar.markdown("### About Agent Systems")
st.sidebar.markdown("""
**Single Agent System:**
- One agent handles the entire workflow
- Simpler architecture
- Potentially faster for simple tasks

**Multi-Agent System:**
- Specialized agents for each phase
- More modular approach
- May handle complex analyses better
""")

# Basic run details expander
if st.sidebar.button("Show Run Details"):
    st.subheader("Run Information")
    
    if agent_mode == "Single Agent" and hasattr(single_analytics, 'last_run') and single_analytics.last_run:
        st.markdown("### Single Agent Last Run")
        st.json(single_analytics.last_run)
    
    if agent_mode == "Multi-Agent" and hasattr(multi_analytics, 'last_run') and multi_analytics.last_run:
        st.markdown("### Multi-Agent Last Run")
        st.json(multi_analytics.last_run)

# Add a footer
st.markdown("---")
st.markdown("Data Science Analysis Tool powered by OpenAI Agent SDK")
st.markdown("*Basic run analytics displayed in sidebar*")