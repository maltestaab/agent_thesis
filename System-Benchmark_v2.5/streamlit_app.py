"""
streamlit_app.py - Main Web Interface for Data Science Analysis Tool

This is the primary user interface built with Streamlit that provides:
- File upload and data preview functionality
- Real-time streaming analysis with live progress updates
- Support for both single-agent and multi-agent analysis modes
- Live analytics display (duration, tool calls, costs, images)
- Interactive controls for starting, stopping, and configuring analysis

The interface handles all user interactions and coordinates with the agent systems
to provide a seamless data science analysis experience.
"""

import streamlit as st
import asyncio
import os
import pandas as pd
import time
import re
from dotenv import load_dotenv
import nest_asyncio
from collections import deque

# Enable nested asyncio loops for Streamlit compatibility
nest_asyncio.apply()

# Load environment variables for API keys and configuration
load_dotenv()

# Import the core analysis systems
from data_science_agents.agent_systems.single_agent import run_single_agent_analysis
from data_science_agents.agent_systems.multi_agent import run_multi_agent_analysis
from data_science_agents.core.execution import get_created_images
from data_science_agents.config.settings import SUPPORTED_FILE_TYPES
from data_science_agents.config.prompts import ANALYSIS_PROMPT_TEMPLATE


def display_images_gallery(images):
    """
    Display analysis-generated visualizations in a user-friendly gallery format.
    
    This function takes a list of image file paths and displays them in a two-column
    layout within the Streamlit interface. It filters for valid image files and
    handles display errors gracefully.
    
    Args:
        images (list): List of file paths to images created during analysis
    """
    if not images:
        return
    
    # Define supported image file extensions for display
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff', '.webp')
    
    # Filter to only include valid image files that actually exist
    image_files = [img for img in images if img.lower().endswith(image_extensions) and os.path.exists(img)]
    
    if not image_files:
        return
    
    # Create gallery section header
    st.subheader("📸 Generated Visualizations")
    
    # Display images in two-column layout for better presentation
    cols = st.columns(2)
    for i, img_path in enumerate(image_files):
        try:
            with cols[i % 2]:  # Alternate between left and right columns
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        except Exception as e:
            # Skip files that can't be displayed as images
            continue


def create_streaming_ui():
    """
    Create the real-time streaming interface containers for live analysis updates.
    
    This function sets up the UI components that display live progress during analysis:
    - Agent output: Shows the AI agent's reasoning and text output in real-time
    - Tool activity: Displays when agents execute code or use tools
    - Event log: Provides a timeline of analysis events
    
    Returns:
        tuple: (agent_output, tool_activity, event_log) - Streamlit containers for live updates
    """
    # Create main container for all streaming content
    streaming_container = st.container()
    
    with streaming_container:
        st.subheader("Analysis Progress")
        
        # Create two-column layout: main output and activity sidebar
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Left column: Agent's main reasoning and analysis output
            st.markdown("**Agent Output:**")
            agent_output = st.empty()  # Placeholder for live text updates
            
        with col2:
            # Right column: Tool usage and activity notifications
            st.markdown("**Tool Activity:**")
            tool_activity = st.empty()  # Placeholder for tool execution updates
        
        # Bottom section: Event timeline for tracking analysis progress
        st.markdown("**Event Log:**")
        event_log = st.empty()  # Placeholder for event history
    
    return agent_output, tool_activity, event_log


## MAIN APPLICATION INTERFACE ##

# Configure Streamlit page settings and layout
st.set_page_config(
    page_title="Data Science Analysis Tool",
    layout="wide",  # Use full screen width
    initial_sidebar_state="expanded",  # Show sidebar by default
    page_icon="📊"
)

# Custom CSS styling for the streaming interface components
st.markdown("""
<style>
    .streaming-text {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        max-height: 400px;
        overflow-y: auto;
    }
    .tool-activity {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .event-log {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.8rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables for maintaining application state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

if 'cancel_analysis' not in st.session_state:
    st.session_state.cancel_analysis = False

if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None

# Main application header and description
st.title("📊 Data Science Analysis Tool")
st.markdown("Upload your data and analyze it using AI agents with **live streaming**")

# Sidebar configuration panel
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Agent mode selection: Single vs Multi-Agent
    agent_mode = st.radio(
        "Analysis Mode",
        ["Single Agent", "Multi-Agent"],
        help="Single Agent: One AI handles all phases. Multi-Agent: Specialized agents for different phases."
    )
    
    # Toggle for live streaming display
    show_live_streaming = st.checkbox(
        "🔴 Show Live Progress",
        value=True,
        help="Show real-time progress with live agent output and tool calls"
    )
    
    st.markdown("---")
    
    # Model selection and configuration
    st.subheader("🤖 Model Settings")
    
    # Available AI models with different capabilities and costs
    available_models = [
        "gpt-4o-mini",  # Fast and economical
        "gpt-4o",       # Latest and most capable
        "gpt-4",        # Previous generation, still very capable
        "gpt-3.5-turbo" # Fastest and most economical
    ]
    
    # Model descriptions to help users choose
    model_descriptions = {
        "gpt-4o-mini": "Fast and cost-effective, ideal for most data analysis tasks",
        "gpt-4o": "Latest GPT-4 model with best performance and reasoning capabilities", 
        "gpt-4": "Previous GPT-4 version, very capable for complex analysis",
        "gpt-3.5-turbo": "Fastest and most economical option for basic analysis"
    }
    
    # Model selection dropdown
    selected_model = st.selectbox(
        "Choose AI Model",
        available_models,
        index=0,  # Default to gpt-4o-mini for cost-effectiveness
        help="Select the AI model for analysis"
    )
    
    # Display model information and pricing
    if selected_model:
        from data_science_agents.config.settings import TOKEN_COSTS
        costs = TOKEN_COSTS[selected_model]
        
        # Show model description
        st.markdown(f"**{model_descriptions[selected_model]}**")
        
        # Display pricing information in a clean format
        st.markdown("**Pricing per 1 million tokens:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📥 **Input:** ${costs['input']:.2f}")
        with col2:
            st.write(f"📤 **Output:** ${costs['output']:.2f}")

# Main content area: Data upload and analysis
st.header("📁 Data Upload & Analysis")

# File upload widget for datasets
uploaded_file = st.file_uploader(
    "Choose your dataset",
    type=SUPPORTED_FILE_TYPES,
    help="Upload a CSV or Excel file containing your data"
)

# Immediate data preview when file is uploaded
if uploaded_file:
    try:
        # Load data based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display file information
        st.success(f"✅ **{uploaded_file.name}** - {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Data preview section
        st.subheader("📋 Dataset Preview")
        
        # Key statistics in a compact metric layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Display first 10 rows of data
        st.dataframe(df.head(10), use_container_width=True)
        
        # Detailed column information in expandable section
        with st.expander("📊 Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Reset file pointer for analysis use
        uploaded_file.seek(0)
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Analysis request section
st.subheader("🎯 Analysis Request")

# Text area for user to describe their analysis needs
user_prompt = st.text_area(
    "Describe your analysis needs",
    value="Please find the effect the explanatory variables have on the target variable winpercent and visualize your results.",
    height=100,
    help="Be specific about what insights you're looking for"
)

# Control buttons for starting and stopping analysis
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    # Stop button (only enabled when analysis is running)
    stop_analysis = st.button(
        "🛑 Stop Analysis",
        disabled=not st.session_state.analysis_running,
        use_container_width=True
    )
    if stop_analysis:
        st.session_state.cancel_analysis = True
        st.warning("⏹️ Cancellation requested...")

with col2:
    # Main analysis button (disabled when running or missing requirements)
    run_analysis = st.button(
        f"🚀 Run {agent_mode} Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.analysis_running or not uploaded_file or not user_prompt
    )

with col3:
    st.empty()  # Spacing column

# Handle analysis start
if run_analysis:
    # Set running state and reset cancellation flag
    st.session_state.analysis_running = True
    st.session_state.cancel_analysis = False
    st.session_state.analytics_data = None
    
    # Store analysis configuration in session state
    st.session_state.file_name = uploaded_file.name
    st.session_state.user_prompt = user_prompt
    st.session_state.agent_mode = agent_mode
    st.session_state.show_live_streaming = show_live_streaming
    st.session_state.selected_model = selected_model  
    
    # Save uploaded file to working directory for analysis
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Force page refresh to update UI state
    st.rerun()

# Main analysis execution block
if st.session_state.analysis_running and 'file_name' in st.session_state:
    # Create formatted analysis prompt for the AI agents
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        file_name=st.session_state.file_name,
        user_prompt=st.session_state.user_prompt
    )
    
    # === LIVE ANALYTICS DISPLAY ===
    st.markdown("---")
    
    # Create live analytics metrics at the top
    st.subheader("📊 Analytics")
    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
    with analytics_col1:
        duration_metric = st.empty()  # Live duration counter
    with analytics_col2:
        tools_metric = st.empty()     # Tool usage counter
    with analytics_col3:
        cost_metric = st.empty()      # Cost estimation
    with analytics_col4:
        images_metric = st.empty()    # Image creation counter
    
    st.markdown("---")
    
    # Create streaming interface if user wants live progress
    if st.session_state.show_live_streaming:
        agent_output, tool_activity, event_log = create_streaming_ui()
    
    def update_live_analytics_from_event(analytics_content: str):
        """
        Parse analytics events from agent systems and update the live display.
        
        Agent systems send periodic analytics updates in a structured format.
        This function parses those updates and refreshes the metrics display.
        
        Args:
            analytics_content (str): Formatted analytics data from agent systems
        """
        try:
            # Parse format: "ANALYTICS_UPDATE|duration|tool_calls|cost|images"
            parts = analytics_content.split("|")
            if len(parts) >= 5 and parts[0] == "ANALYTICS_UPDATE":
                duration = float(parts[1])
                tool_calls = int(parts[2]) 
                cost = float(parts[3])
                images = int(parts[4])
                
                # Update the live metrics display
                duration_metric.metric("Duration", f"{duration:.1f}s")
                tools_metric.metric("Tool Calls", tool_calls)
                cost_metric.metric("Est. Cost", f"${cost:.4f}")
                images_metric.metric("Images", images)
        except:
            # Ignore parsing errors to maintain stability
            pass
    
    async def run_streaming_analysis():
        """
        Main analysis execution function with real-time streaming.
        
        This function coordinates the entire analysis process:
        1. Starts the appropriate agent system (single or multi-agent)
        2. Processes streaming events for live UI updates
        3. Handles user cancellation and error conditions
        4. Displays final results and created visualizations
        """
        
        try:
            # Choose analysis approach based on user selection
            if st.session_state.agent_mode == "Single Agent":
                analysis_stream = run_single_agent_analysis(
                    analysis_prompt, 
                    st.session_state.file_name, 
                    max_turns=50,
                    model=st.session_state.get('selected_model', 'gpt-4o-mini')
                )
            else:
                analysis_stream = run_multi_agent_analysis(
                    analysis_prompt, 
                    st.session_state.file_name, 
                    max_turns=50,
                    model=st.session_state.get('selected_model', 'gpt-4o-mini')
                )
            
            # Initialize streaming variables for live display
            if st.session_state.show_live_streaming:
                agent_text = ""           # Current visible text in agent output
                full_agent_text = ""      # Complete conversation history
                text_buffer = ""          # Buffer for batching text updates
                last_text_update = time.time()
                
                full_conversation = []    # Complete conversation log
                tool_events = deque()     # Tool activity timeline
                event_history = deque()   # Event history for debugging
                
                # Timing controls for UI update frequency
                last_tool_update = time.time()
                event_counter = 0
            
            # Main event processing loop
            async for event in analysis_stream:
                current_time = time.time()
                
                # === ANALYTICS PROCESSING ===
                # Update live analytics display when agents send updates
                if event.event_type == "analytics_update":
                    update_live_analytics_from_event(event.content)
                
                # === LIVE STREAMING DISPLAY ===
                # Process events for real-time UI updates
                if st.session_state.show_live_streaming:
                    if event.event_type == "text_delta":
                        # Handle incremental text updates (like ChatGPT typing effect)
                        text_buffer += event.content
                        full_agent_text += event.content
                        
                        # Add agent identification for multi-agent clarity
                        agent_prefix = ""
                        if "Agent" in event.agent_name and event.agent_name != "Agent":
                            agent_prefix = f"[{event.agent_name}] "
                        
                        # Update display periodically for performance
                        if current_time - last_text_update >= 2.5:
                            agent_text += agent_prefix + text_buffer
                            text_buffer = ""
                            
                            # Limit visible text length for performance
                            if len(agent_text) > 500:
                                agent_text = agent_text[-500:]
                            
                            agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                            last_text_update = current_time
                    
                    elif event.event_type == "tool_call":
                        # Display tool execution notifications
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 2.5:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "tool_output":
                        # Display tool completion notifications
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - ✅ Execution completed")
                        
                        if current_time - last_tool_update >= 2.5:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "specialist_start":
                        # Multi-agent: Specialist agent starting work
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 1.0:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "specialist_complete":
                        # Multi-agent: Specialist agent finished work
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 1.0:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "agent_reasoning": 
                        # Display agent reasoning for multi-agent mode
                        if st.session_state.agent_mode == "Multi-Agent":
                            full_agent_text += f"\n🤔 {event.agent_name}: {event.content}\n"

                    elif event.event_type in ["agent_handoff", "agent_result", "sub_agent_start", "sub_agent_complete"]:
                        # Handle agent coordination events
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 2.5:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "message_complete":
                        # Flush any remaining text buffer at message completion
                        if text_buffer:
                            agent_text += text_buffer
                            full_agent_text += text_buffer
                            text_buffer = ""
                            
                            if len(agent_text) > 500:
                                agent_text = agent_text[-500:]
                            
                            agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                        
                        # Reset for next message
                        agent_text = ""
                    
                    # Update event history for debugging and monitoring
                    if event.event_type not in ["text_delta", "analytics_update"]:
                        event_display = event.event_type
                        if event.event_type in ["specialist_start", "specialist_complete"]:
                            event_display = f"{event.event_type} ({event.agent_name})"
                        
                        event_history.append(f"[{event.agent_name}] {event_display}")
                        event_counter += 1
                        
                        # Update event log periodically
                        if event_counter % 5 == 0:
                            recent_events = list(event_history)[-4:]
                            event_log.markdown(f'<div class="event-log">{"<br>".join(recent_events)}</div>', unsafe_allow_html=True)
                    
                    # Small delay for smooth streaming effect
                    await asyncio.sleep(0.05)
                
                # === COMPLETION AND ERROR HANDLING ===
                if event.event_type == "analysis_complete":
                    # Analysis completed successfully
                    st.session_state.analysis_running = False
                    
                    # Flush any remaining text buffer
                    if st.session_state.show_live_streaming and 'text_buffer' in locals() and text_buffer:
                        agent_text += text_buffer
                        full_agent_text += text_buffer
                        if len(agent_text) > 600:
                            agent_text = agent_text[-600:]
                        agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                    
                    # Convert live displays to final scrollable versions
                    if st.session_state.show_live_streaming:
                        if full_agent_text or full_conversation:
                            combined_content = ""
                            if full_agent_text:
                                combined_content += full_agent_text
                            if full_conversation:
                                if combined_content:
                                    combined_content += "<br><br>" + "<br><br>".join(full_conversation)
                                else:
                                    combined_content = "<br><br>".join(full_conversation)
                            agent_output.markdown(f'<div class="streaming-text">{combined_content}</div>', unsafe_allow_html=True)
                        
                        if tool_events:
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(tool_events)}</div>', unsafe_allow_html=True)
                        
                        if event_history:
                            event_log.markdown(f'<div class="event-log">{"<br>".join(event_history)}</div>', unsafe_allow_html=True)
                    
                    st.success("🎉 Analysis completed!")
                    
                    # Display final analysis results
                    st.markdown("### 📊 Final Analysis Results")
                    st.markdown(event.content)
                    
                    # Display any visualizations created during analysis
                    images = get_created_images()
                    if images:
                        time.sleep(0.3)  # Brief pause for UI stability
                        st.markdown("---")
                        display_images_gallery(images)
                    
                    return
                    
                elif event.event_type == "analysis_error":
                    # Analysis failed with error
                    st.session_state.analysis_running = False
                    st.error(event.content)
                    return
                    
                elif event.event_type == "analysis_cancelled":
                    # Analysis cancelled by user
                    st.session_state.analysis_running = False
                    st.warning(event.content)
                    return
                    
        except Exception as e:
            # Handle unexpected errors during analysis
            st.error(f"Streaming analysis failed: {str(e)}")
            st.session_state.analysis_running = False
    
    # Execute the analysis with error handling
    try:
        asyncio.run(run_streaming_analysis())
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        st.session_state.analysis_running = False
        st.session_state.cancel_analysis = False