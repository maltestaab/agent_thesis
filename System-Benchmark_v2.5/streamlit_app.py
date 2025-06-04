"""
streamlit_app.py - Main Web Interface with Specialist Streaming Support

STREAMING SPECIALISTS:
- Handles real-time events from both orchestrator and specialist agents
- Shows specialist reasoning, code execution, and results as they happen
- Provides complete transparency into multi-agent workflow
- Maintains all simplifications while adding full specialist visibility
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

# Apply nest_asyncio to handle asyncio in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

from data_science_agents.agent_systems.single_agent import run_single_agent_analysis
from data_science_agents.agent_systems.multi_agent import run_multi_agent_analysis
from data_science_agents.core.execution import get_created_images
from data_science_agents.config.settings import SUPPORTED_FILE_TYPES
from data_science_agents.config.prompts import ANALYSIS_PROMPT_TEMPLATE


def display_images_gallery(images):
    """Display created visualizations in a gallery format."""
    if not images:
        return
    
    # Valid image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff', '.webp')
    
    # Filter for image files only
    image_files = [img for img in images if img.lower().endswith(image_extensions) and os.path.exists(img)]
    
    if not image_files:
        return
    
    st.subheader("📸 Generated Visualizations")
    
    # Display images in two columns for better layout
    cols = st.columns(2)
    for i, img_path in enumerate(image_files):
        try:
            with cols[i % 2]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        except Exception as e:
            # Skip files that can't be displayed as images
            continue
                
def create_streaming_ui():
    """Create the real-time interface containers for live analysis updates."""
    
    # Main containers for all streaming updates
    streaming_container = st.container()
    
    with streaming_container:
        st.subheader("Analysis Progress")
        
        # Create columns for different types of updates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Agent messages streaming output
            st.markdown("**Agent Output:**")
            agent_output = st.empty()
            
        with col2:
            # Tool calls and status updates
            st.markdown("**Tool Activity:**")
            tool_activity = st.empty()
        
        # Event log at bottom
        st.markdown("**Event Log:**")
        event_log = st.empty()
    
    return agent_output, tool_activity, event_log


## MAIN APP ##

# Page configuration
st.set_page_config(
    page_title="Data Science Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# CSS styling for the streaming UI
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

# Initialize session state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

if 'cancel_analysis' not in st.session_state:
    st.session_state.cancel_analysis = False

if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None

# Main header
st.title("📊 Data Science Analysis Tool")
st.markdown("Upload your data and analyze it using AI agents with **live streaming**")

# Sidebar configuration  
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Agent mode selection
    agent_mode = st.radio(
        "Analysis Mode",
        ["Single Agent", "Multi-Agent"],
        help="Single Agent: One AI handles all phases. Multi-Agent: Specialized agents for different phases."
    )
    
    # Streaming toggle
    show_live_streaming = st.checkbox(
        "🔴 Show Live Progress",
        value=True,
        help="Show real-time progress with live agent output and tool calls"
    )
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 Model Settings")
    
    # Available models from token costs
    available_models = [
        "gpt-4o-mini",
        "gpt-4o", 
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    # Model descriptions
    model_descriptions = {
        "gpt-4o-mini": "Fast and cost-effective, ideal for most data analysis tasks",
        "gpt-4o": "Latest GPT-4 model with best performance and reasoning capabilities", 
        "gpt-4": "Previous GPT-4 version, very capable for complex analysis",
        "gpt-3.5-turbo": "Fastest and most economical option for basic analysis"
    }
    
    selected_model = st.selectbox(
        "Choose AI Model",
        available_models,
        index=0,  # Default to gpt-4o-mini
        help="Select the AI model for analysis"
    )
    
    # Show model description and pricing
    if selected_model:
        from data_science_agents.config.settings import TOKEN_COSTS
        costs = TOKEN_COSTS[selected_model]
        
        # Description
        st.markdown(f"**{model_descriptions[selected_model]}**")
        
        # Pricing in a clean, smaller format
        st.markdown("**Pricing per 1 million tokens:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📥 **Input:** ${costs['input']:.2f}")
        with col2:
            st.write(f"📤 **Output:** ${costs['output']:.2f}")

# Main layout
st.header("📁 Data Upload & Analysis")

# File upload section
uploaded_file = st.file_uploader(
    "Choose your dataset",
    type=SUPPORTED_FILE_TYPES,
    help="Upload a CSV or Excel file containing your data"
)

# Immediate data preview when file is uploaded
if uploaded_file:
    try:
        # Load and preview data immediately
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # File info in compact format
        st.success(f"✅ **{uploaded_file.name}** - {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Immediate data preview
        st.subheader("📋 Dataset Preview")
        
        # Key stats in compact format
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Data preview table
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information in expandable section
        with st.expander("📊 Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Reset file pointer for analysis
        uploaded_file.seek(0)
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Analysis request section
st.subheader("🎯 Analysis Request")

# Analysis prompt input
user_prompt = st.text_area(
    "Describe your analysis needs",
    value="Please find the effect the explanatory variables have on the target variable winpercent and visualize your results.",
    height=100,
    help="Be specific about what insights you're looking for"
)

# Analysis buttons with stop functionality
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    stop_analysis = st.button(
        "🛑 Stop Analysis",
        disabled=not st.session_state.analysis_running,
        use_container_width=True
    )
    if stop_analysis:
        st.session_state.cancel_analysis = True
        st.warning("⏹️ Cancellation requested...")

with col2:
    run_analysis = st.button(
        f"🚀 Run {agent_mode} Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.analysis_running or not uploaded_file or not user_prompt
    )

with col3:
    st.empty()  # Spacing

# Run analysis when button is clicked
if run_analysis:
    st.session_state.analysis_running = True
    st.session_state.cancel_analysis = False
    st.session_state.analytics_data = None
    
    # Save uploaded file and analysis details
    st.session_state.file_name = uploaded_file.name
    st.session_state.user_prompt = user_prompt
    st.session_state.agent_mode = agent_mode
    st.session_state.show_live_streaming = show_live_streaming
    st.session_state.selected_model = selected_model  
    
    # Save the file to working directory
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Force page refresh to show the stop button as enabled
    st.rerun()

# Run analysis if we're in running state
if st.session_state.analysis_running and 'file_name' in st.session_state:
    # Create analysis prompt
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        file_name=st.session_state.file_name,
        user_prompt=st.session_state.user_prompt
    )
    
    # === STREAMING ANALYSIS ===
    st.markdown("---")
    
    # Create live analytics display at the top
    st.subheader("📊 Analytics")
    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
    with analytics_col1:
        duration_metric = st.empty()
    with analytics_col2:
        tools_metric = st.empty()
    with analytics_col3:
        cost_metric = st.empty()
    with analytics_col4:
        images_metric = st.empty()
    
    st.markdown("---")
    
    # Create streaming UI containers only if user wants to see live progress
    if st.session_state.show_live_streaming:
        agent_output, tool_activity, event_log = create_streaming_ui()
    
    def update_live_analytics_from_event(analytics_content: str):
        """
        Parse analytics event from agent systems and update display.
        
        This maintains our simplifications - agent systems track analytics
        using our helpers, and share the data via events. No duplicate tracking.
        """
        try:
            # Parse format: "ANALYTICS_UPDATE|45.2|8|0.0142|3"
            parts = analytics_content.split("|")
            if len(parts) >= 5 and parts[0] == "ANALYTICS_UPDATE":
                duration = float(parts[1])
                tool_calls = int(parts[2]) 
                cost = float(parts[3])
                images = int(parts[4])
                
                # Update the live display
                duration_metric.metric("Duration", f"{duration:.1f}s")
                tools_metric.metric("Tool Calls", tool_calls)
                cost_metric.metric("Est. Cost", f"${cost:.4f}")
                images_metric.metric("Images", images)
        except:
            pass  # Ignore parsing errors
    
    async def run_streaming_analysis():
        """
        SPECIALIST STREAMING: Handles events from both orchestrator and specialists.
        Shows real-time specialist work with complete transparency.
        """
        
        try:
            # Choose analysis function based on mode
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
            
            # Initialize streaming variables only if showing live progress
            if st.session_state.show_live_streaming:
                agent_text = ""
                full_agent_text = ""
                text_buffer = ""
                last_text_update = time.time()
                
                full_conversation = []
                tool_events = deque()
                event_history = deque()
                
                # Time-based update tracking
                last_tool_update = time.time()
                event_counter = 0
            
            # Main event loop for streaming analysis
            async for event in analysis_stream:
                current_time = time.time()
                
                # === PARSE ANALYTICS EVENTS (NO manual tracking) ===
                if event.event_type == "analytics_update":
                    update_live_analytics_from_event(event.content)
                
                # Process streaming events if live streaming is enabled
                if st.session_state.show_live_streaming:
                    if event.event_type == "text_delta":
                        # Buffer text updates for performance
                        text_buffer += event.content
                        full_agent_text += event.content
                        
                        # Show which agent is talking
                        agent_prefix = ""
                        if "Agent" in event.agent_name and event.agent_name != "Agent":
                            agent_prefix = f"[{event.agent_name}] "
                        
                        if current_time - last_text_update >= 2.5:
                            agent_text += agent_prefix + text_buffer
                            text_buffer = ""
                            
                            # Limit live visible text to 500 characters for better performance
                            if len(agent_text) > 500:
                                agent_text = agent_text[-500:]
                            
                            agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                            last_text_update = current_time
                    
                    elif event.event_type == "tool_call":
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 2.5:
                            recent_tools = list(tool_events)[-3:]  # Show more for specialists
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "tool_output":
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - ✅ Execution completed")
                        
                        if current_time - last_tool_update >= 2.5:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "specialist_start":
                        # Specialist agent starting work
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 1.0:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "specialist_complete":
                        # Specialist agent finished work
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 1.0:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "agent_reasoning": 
                        if st.session_state.agent_mode == "Multi-Agent":
                            full_agent_text += f"\n🤔 {event.agent_name}: {event.content}\n"

                    elif event.event_type in ["agent_handoff", "agent_result", "sub_agent_start", "sub_agent_complete"]:
                        tool_events.append(f"⏰ {time.strftime('%H:%M:%S')} - {event.content}")
                        
                        if current_time - last_tool_update >= 2.5:
                            recent_tools = list(tool_events)[-3:]
                            tool_activity.markdown(f'<div class="tool-activity">{"<br>".join(recent_tools)}</div>', unsafe_allow_html=True)
                            last_tool_update = current_time
                    
                    elif event.event_type == "message_complete":
                        # Flush any remaining text buffer
                        if text_buffer:
                            agent_text += text_buffer
                            full_agent_text += text_buffer
                            text_buffer = ""
                            
                            if len(agent_text) > 500:
                                agent_text = agent_text[-500:]
                            
                            agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                        
                        # Reset for next message
                        agent_text = ""
                    
                    # Add events to history
                    if event.event_type not in ["text_delta", "analytics_update"]:
                        event_display = event.event_type
                        if event.event_type in ["specialist_start", "specialist_complete"]:
                            event_display = f"{event.event_type} ({event.agent_name})"
                        
                        event_history.append(f"[{event.agent_name}] {event_display}")
                        event_counter += 1
                        
                        if event_counter % 5 == 0:  # More frequent updates for specialists
                            recent_events = list(event_history)[-4:]
                            event_log.markdown(f'<div class="event-log">{"<br>".join(recent_events)}</div>', unsafe_allow_html=True)
                    
                    # Small delay for streaming visibility
                    await asyncio.sleep(0.05)
                
                # Handle completion or error
                if event.event_type == "analysis_complete":
                    st.session_state.analysis_running = False
                    
                    # Flush any remaining text buffer
                    if st.session_state.show_live_streaming and 'text_buffer' in locals() and text_buffer:
                        agent_text += text_buffer
                        full_agent_text += text_buffer
                        if len(agent_text) > 600:
                            agent_text = agent_text[-600:]
                        agent_output.markdown(f'<div class="streaming-text">{agent_text}</div>', unsafe_allow_html=True)
                    
                    # Convert live windows to full scrollable versions
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
                    
                    # Show final results
                    st.markdown("### 📊 Final Analysis Results")
                    st.markdown(event.content)
                    
                    # Show images using single source from execution.py
                    images = get_created_images()
                    if images:
                        time.sleep(0.3)
                        st.markdown("---")
                        display_images_gallery(images)
                    
                    return
                    
                elif event.event_type == "analysis_error":
                    st.session_state.analysis_running = False
                    st.error(event.content)
                    return
                    
                elif event.event_type == "analysis_cancelled":
                    st.session_state.analysis_running = False
                    st.warning(event.content)
                    return
                    
        except Exception as e:
            st.error(f"Streaming analysis failed: {str(e)}")
            st.session_state.analysis_running = False
    
    # Run streaming analysis
    try:
        asyncio.run(run_streaming_analysis())
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        st.session_state.analysis_running = False
        st.session_state.cancel_analysis = False